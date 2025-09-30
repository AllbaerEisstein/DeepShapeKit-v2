from typing import Optional

import torch
from src import fish_model_edit as fish_model
from src.Silhouette_Renderer_edit import Silhouette_Renderer
from src.losses_edit import keypoint_reprojection_loss_global, kpt_repr_plus_bone_pose_and_length_loss, mask_fitting_loss


class OptimizeMV:
    """
    Perform multi-view mesh fitting in three stages:
      1. Global rotation, translation, scale
      2. Body-pose and bone-length refinement
      3. Tail and silhouette offset refinement

    On call, returns posed vertices, pose parameters, bone lengths,
    scale, translation, plus a placeholder tuple.
    """

    def __init__(
        self,
        fish_model_obj: fish_model.fish_model,
        lim_weight=1,
        prior_weight=1,
        bone_weight=1,
        mask_weight=1,
        smooth_weights=None,
        step_size=1e-2,
        num_iters=100,
        device=torch.device("cpu"),
    ):
        # Store hyper-parameters
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.lim_weight = lim_weight
        self.prior_weight = prior_weight
        self.bone_weight = bone_weight
        self.mask_weight = mask_weight
        self.smooth_weights = smooth_weights or [1.0, 1.0, 1.0]

        # Load parametric fish mesh and faces
        self.fish = fish_model_obj
        self.faces = self.fish.faces

    def __call__(
        self,
        init_ori_plus_pose: torch.Tensor,
        init_body_bone_length: torch.Tensor,
        init_t: torch.Tensor,
        init_scale: torch.Tensor,
        proj_m: torch.Tensor,
        keypoints: torch.Tensor,
        masks: torch.Tensor,
        silhouette_renderer: Silhouette_Renderer,
        has_prev: bool = False,
        index: Optional[torch.Tensor] = None,
        bboxes: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            init_ori_plus_pose (1, 3 + bn*3): [global_orient + body_pose]
            init_body_bone_length (1, bn): body bone lengths
            init_t (1, 3): translation
            init_scale (1,): scale factor
            proj_m (vn, 3, 4): projection matrices for vn views
            keypoints (vn, kn, 3): 2D keypoints + confidence
            masks (vn, H, W): silhouette masks
        """
        if not (
            all(
                self.device == param.device
                for param in [
                    keypoints,
                    masks,
                    self.fish,
                    init_ori_plus_pose,
                    init_body_bone_length,
                    init_t,
                    init_scale,
                    proj_m,
                ]
            )
            and self.fish.device_active == True
        ):
            keypoints = keypoints.to(self.device)
            masks = masks.to(self.device) 
            self.fish.to_device(self.device) 
            init_ori_plus_pose = init_ori_plus_pose.to(self.device) 
            init_body_bone_length = init_body_bone_length.to(self.device) 
            init_t = init_t.to(self.device) 
            init_scale = init_scale.to(self.device) 
            proj_m = proj_m.to(self.device) 
            
        if not "cuda" in str(keypoints.device):
            print(
                f"\nWarning: running pose optimization on device {str(keypoints.device)}\n"
            )

        # ===== Prepare data =====
        batch_size = proj_m.shape[0]
        kpts_2d = keypoints[..., :2]
        kpts_conf = keypoints[..., 2].clone()


        # ===== Initialize parameters =====
        # global_orient: (1,3), body_pose: (1,P), bone_length: (1,B)
        global_orient = init_ori_plus_pose[:, :3].clone().detach()
        body_pose = init_ori_plus_pose[:, 3:].clone().detach()
        global_t = init_t.clone().detach()
        body_bone_length = init_body_bone_length.clone().detach()
        scale = init_scale.clone().detach()

        # ============ Stage 1: optimize global_orient, translation, scale ============
        # ============ loss: keypoint reprojection l2 distance, mask l1 distance ======
        for param in [body_pose, body_bone_length]:
            param.requires_grad_(False)
        for param in [global_orient, global_t, scale]:
            param.requires_grad_(True)
        opt_global = torch.optim.Adam(
            [global_orient, global_t, scale], lr=self.step_size
        )
        for _ in range(self.num_iters):
            print(f"Stage 1 - global ori: {global_orient}")
            print(f"Stage 1 - body global t: {global_t}")
            print(f"Stage 1 - scale: {scale}")
            print(f"Stage 1 - body pose: {body_pose}")
            print(f"Stage 1 - body bone length: {body_bone_length}")
            # Apply global ori and scale
            out = self.fish(
                global_ori=global_orient,
                body_pose=body_pose,
                body_bone_length=body_bone_length,
                scale=scale,
            )
            # Apply global t
            model_kpts = out["keypoints"].to(self.device) + global_t
            # Keypoint reprojection loss
            model_kpts = model_kpts.expand(batch_size, -1, -1)
            # TODO: filter missing keypoints before this step based on confidence
            # -> in kpt reprojection loss, squared confidence is a factor for each keypoint loss
            loss = (
                keypoint_reprojection_loss_global(model_kpts, proj_m, kpts_2d, kpts_conf)
                + self.prior_weight * (global_t - init_t).abs().sum()
            )
            # Silhouette loss
            silhouette_renders = silhouette_renderer(
                out["vertices"], self.faces.unsqueeze(0), global_t
            )
            loss += mask_fitting_loss(
                silhouette_renders, masks.float(), 0.1 * self.mask_weight
            )
            opt_global.zero_grad()
            loss.backward()
            opt_global.step()

        # ============ Stage 2: refine body_pose, bone_length, global_t, scale ============
        # ============ loss: keypoint L2-distance, bone constraint loss (angle/length min and max)
        for param in [body_pose, body_bone_length, global_orient, global_t, scale]:
            param.requires_grad_(True)
        opt_body = torch.optim.Adam(
            [body_pose, body_bone_length, global_orient, global_t, scale], lr=self.step_size
        )
        # relax tail keypoints
        kpts_conf = kpts_conf.fill_(0.8)
        # TODO: what happens here?
        kpts_conf[:, -3] = 0
        kpts_conf[:, -1] = 0
        # TODO: is this disabling y == 0 keypoints?
        kpts_conf[keypoints[..., 2] == 0] = 0
        for _ in range(self.num_iters):
            print(f"Stage 2 - global ori: {global_orient}")
            print(f"Stage 2 - body global t: {global_t}")
            print(f"Stage 2 - scale: {scale}")
            print(f"Stage 2 - body pose: {body_pose}")
            print(f"Stage 2 - body bone length: {body_bone_length}")
            out = self.fish(
                global_ori=global_orient,
                body_pose=body_pose,
                body_bone_length=body_bone_length,
                scale=scale,
            )
            m_kpts = out["keypoints"].to(self.device) + global_t
            m_kpts = m_kpts.expand(batch_size, -1, -1)
            loss = kpt_repr_plus_bone_pose_and_length_loss(
                m_kpts,
                self.fish.bone_angle_min,
                self.fish.bone_angle_max,
                self.fish.bone_length_min,
                self.fish.bone_length_max,
                proj_m,
                kpts_2d,
                kpts_conf,
                body_pose,
                body_bone_length,
                lim_weight=self.lim_weight,
                prior_weight=self.prior_weight,
                bone_weight=self.bone_weight,
            )
            opt_body.zero_grad()
            loss.backward()
            opt_body.step()


        # ============ Stage 3: tail + silhouette offset ============
        # TODO: silhouette offset?
        sil_offset = torch.zeros((2, 3), device=self.device, requires_grad=True)
        for p in [body_pose, body_bone_length, global_orient, global_t, scale, sil_offset]:
            p.requires_grad_(True)
        opt_tail = torch.optim.Adam(
            [body_pose, body_bone_length, global_orient, global_t, scale, sil_offset],
            lr=self.step_size,
        )
        # reweight tail points
        # TODO: specify which keypoints belong to tail, body, etc. in fish fish model
        kpts_conf = kpts_conf.fill_(0.8)
        kpts_conf[0, -3] = 0.1
        kpts_conf[0, -1] = 1
        kpts_conf[1, 0] = 1
        kpts_conf[1, 2] = 1
        # TODO: Why disable keypoints where y = 0?
        kpts_conf[keypoints[..., 2] == 0] = 0
        init_bp = body_pose.clone().detach()
        init_bl = body_bone_length.clone().detach()
        for _ in range(self.num_iters):
            print(f"Stage 3 - global ori: {global_orient}")
            print(f"Stage 3 - body global t: {global_t}")
            print(f"Stage 3 - scale: {scale}")
            print(f"Stage 3 - body pose: {body_pose}")
            print(f"Stage 3 - body bone length: {body_bone_length}")
            out = self.fish(
                global_ori=global_orient,
                body_pose=body_pose,
                body_bone_length=body_bone_length,
                scale=scale,
            )
            m_kpts = out["keypoints"].to(self.device) + global_t
            m_kpts = m_kpts.expand(batch_size, -1, -1)
            loss = kpt_repr_plus_bone_pose_and_length_loss(
                m_kpts,
                self.fish.bone_angle_min,
                self.fish.bone_angle_max,
                self.fish.bone_length_min,
                self.fish.bone_length_max,
                proj_m,
                kpts_2d,
                kpts_conf,
                body_pose,
                body_bone_length,
                lim_weight=self.lim_weight,
                prior_weight=self.prior_weight,
                bone_weight=self.bone_weight,
                pose_init=init_bp,
                bone_init=init_bl,
            )
            opt_tail.zero_grad()
            loss.backward()
            opt_tail.step()

        # Gather final outputs
        vertices = out["vertices"].detach().cpu()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach().cpu()
        bone = body_bone_length.detach().cpu()
        scale = scale.detach().cpu()
        translation = global_t.detach().cpu()
        return vertices, pose, bone, scale, translation, (0, 0)
