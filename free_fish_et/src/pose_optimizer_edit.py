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
        angle_constraint_weight: float = 1.0,
        smooth_weight: float = 1.0,
        bone_length_constraint_weight: float = 1.0,
        mask_weight: float = 1.0,
        step_size: float = 1e-2,
        num_iters: int = 100,
        device=torch.device("cpu"),
    ):
        # Store hyper-parameters
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.angle_constraint_weight = angle_constraint_weight
        self.smooth_weight = smooth_weight
        self.bone_length_constraint_weight = bone_length_constraint_weight
        self.mask_weight = mask_weight

        # Load parametric fish mesh and faces
        self.fish = fish_model_obj
        self.faces = self.fish.faces

        # torch.autograd.set_detect_anomaly(True)

    def __call__(
        self,
        init_ori_plus_pose: torch.Tensor,
        init_body_bone_length: torch.Tensor,
        init_t: torch.Tensor,
        init_scale: torch.Tensor,
        proj_m_from_blworld: torch.Tensor,
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
                    proj_m_from_blworld,
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
            proj_m_from_blworld = proj_m_from_blworld.to(self.device) 
            
        if not "cuda" in str(keypoints.device):
            print(
                f"\nWarning: running pose optimization on device {str(keypoints.device)}\n"
            )

        # ===== Prepare data =====
        batch_size = proj_m_from_blworld.shape[0]
        kpts_2d = keypoints[..., :2]
        kpts_conf = keypoints[..., 2].clone()
        # in the extraction pipeline, undetected kpts have been set to conf=-1
        # in the loss function, conf is a coefficient of the loss -> clamp to 0, 1
        kpts_conf = kpts_conf.clamp(0.0, 1.0)

        def reset_kpts_conf():
            nonlocal kpts_conf
            kpts_conf = keypoints[..., 2].clamp(0.0, 1.0).clone()

        def has_nonfinite_grads(params: list[torch.Tensor]) -> bool:
            for param in params:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    return True
            return False

        for name, tensor in [
            ("init_ori_plus_pose", init_ori_plus_pose),
            ("init_body_bone_length", init_body_bone_length),
            ("init_t", init_t),
            ("init_scale", init_scale),
        ]:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"OptimizeMV received non-finite values in {name}.")


        # ===== Initialize parameters =====
        # global_orient: (1,3), body_pose: (1,B,3), bone_length: (1,B)
        global_orient     = init_ori_plus_pose[:, :3].detach().clone().requires_grad_(True)
        body_pose         = init_ori_plus_pose[:, 3:].detach().clone().view(1, self.fish.n_body_bones, 3)   # requires_grad=False (default)
        body_bone_length  = init_body_bone_length.detach().clone()                                         # requires_grad=False (default)
        global_t          = init_t.detach().clone().requires_grad_(True)
        scale             = init_scale.detach().clone().requires_grad_(True)



        # ============ Stage 1: optimize global_orient, translation, scale ============
        # loss: 
        # -----------------------------------
        # keypoint reprojection l2 distance * keypoint confidence², 
        # mask l1 distance, 
        # global_t smooth loss (difference to init_t)
        # ===================================

        opt_global = torch.optim.Adam(
            [global_orient, global_t, scale], lr=self.step_size
        )
        for _ in range(self.num_iters):
            # print(f"Stage 1 - global ori: {global_orient}")
            # print(f"Stage 1 - body global t: {global_t}")
            # print(f"Stage 1 - scale: {scale}")
            # print(f"Stage 1 - body pose: {body_pose}")
            # print(f"Stage 1 - body bone length: {body_bone_length}")
            # Apply global ori and scale
            out = self.fish(
                global_ori=global_orient,
                body_pose=body_pose.flatten(1),
                body_bone_length=body_bone_length,
                scale=scale,
            )
            # Apply global t
            model_kpts = out["keypoints"].to(self.device) + global_t
            # Keypoint reprojection loss
            model_kpts = model_kpts.expand(batch_size, -1, -1)
            loss = (
                keypoint_reprojection_loss_global(model_kpts, proj_m_from_blworld, kpts_2d, kpts_conf)
                + self.smooth_weight * (global_t - init_t).abs().sum()
            )
            # Silhouette loss
            silhouette_renders = silhouette_renderer(
                out["vertices"], self.faces.unsqueeze(0), global_t
            )
            loss += mask_fitting_loss(
                silhouette_renders, masks.float(), 0.1 * self.mask_weight
            )
            if not torch.isfinite(loss):
                raise ValueError("Error: Stage 1 produced non-finite loss. Stopping Stage 1 early.")
                break

            opt_global.zero_grad()
            loss.backward()
            if has_nonfinite_grads([global_orient, global_t, scale]):
                raise ValueError("Error: Stage 1 produced non-finite gradients. Stopping Stage 1 early.")
            opt_global.step()



        # ============ Stage 2: refine first bone group ============
        # optimize:
        # -----------------------------------
        # body_pose, 
        # bone_length, 
        # global_orient,
        # global_t, 
        # scale
        # --------------
        # loss: 
        # -----------------------------------
        # (keypoint L2-distance * keypoint confidence²) of keypoints of bone group 0,
        # mask l1 distance, 
        # bone constraint loss (angle/length min and max)
        # ===================================

        first_bone_group = self.fish.bone_groups[0]
        # body pose excludes head joint so each bone index is actually one higher
        in_first_bone_group = torch.tensor(
            [(b_idx + 1) in first_bone_group for b_idx in range(body_pose.size(1))],
            dtype=torch.bool, device=self.device
        )
        first_kpt_group = self.fish.keypoint_groups[0]
        not_in_first_kpt_group = torch.tensor(
            [k_idx not in first_kpt_group for k_idx in range(kpts_conf.size(1))],
            dtype=torch.bool, device=self.device
        )

        # disable optimization on non-first-bone-group slices of body pose & bone length tensors
        with torch.no_grad():
            frozen_body_pose = body_pose.clone().detach()
            frozen_body_bone_length = body_bone_length.clone().detach()
        optimizable_body_pose        = body_pose[:, in_first_bone_group, :].detach().clone().requires_grad_(True)
        optimizable_body_bone_length = body_bone_length[:, in_first_bone_group].detach().clone().requires_grad_(True)

        opt_body = torch.optim.Adam(
            [optimizable_body_pose, optimizable_body_bone_length, global_orient, global_t, scale], lr=self.step_size
        )
        # Disable keypoints for this step that don't belong to the first bone group
        kpts_conf[:, not_in_first_kpt_group] = 0
        # reset the confidence of keypoints with conf=0:
        kpts_conf[keypoints[..., 2] == 0] = 0

        def recombine_frozen_and_optimized_tensor(full_frozen: torch.Tensor,
                                                optimizable_slice: torch.Tensor,
                                                mask: torch.Tensor) -> torch.Tensor:
            full = full_frozen.clone()
            if full.dim() == 3:      # e.g., (B, J, 3)
                full[:, mask, :] = optimizable_slice
            elif full.dim() == 2:    # e.g., (B, J)
                full[:, mask] = optimizable_slice
            else:
                raise ValueError(f"Unsupported tensor dim: {full.dim()}")
            return full

        for _ in range(self.num_iters):
            # print(f"Stage 2 - global ori: {global_orient}")
            # print(f"Stage 2 - body global t: {global_t}")
            # print(f"Stage 2 - scale: {scale}")
            # print(f"Stage 2 - body pose: {body_pose}")
            # print(f"Stage 2 - body bone length: {body_bone_length}")
            # print(f"Stage 2 - optimizable body pose: {optimizable_body_pose}")
            # print(f"Stage 2 - optimizable body bone length: {optimizable_body_bone_length}")
            # print(f"Stage 2 - frozen body pose: {frozen_body_pose}")
            # print(f"Stage 2 - frozen body bone length: {frozen_body_bone_length}")
            # print(f"Stage 2 - in first bone group mask: {in_first_bone_group}")
            recombined_body_pose = recombine_frozen_and_optimized_tensor(frozen_body_pose, optimizable_body_pose, in_first_bone_group)
            # print(f"Stage 2 - recombined body pose: {recombined_body_pose}")
            recombined_body_bone_length = recombine_frozen_and_optimized_tensor(frozen_body_bone_length, optimizable_body_bone_length, in_first_bone_group)
            # print(f"Stage 2 - recombined body bone length: {recombined_body_bone_length}")
            out = self.fish(
                global_ori=global_orient,
                body_pose=recombined_body_pose.flatten(1),
                body_bone_length=recombined_body_bone_length,
                scale=scale,
            )
            m_kpts = out["keypoints"].to(self.device) + global_t
            m_kpts = m_kpts.expand(batch_size, -1, -1)

            bone_local_oris = out["body_bone_poses_rest_bone_spaces"].to(self.device)
            bone_local_oris_bone_group = torch.where(
                in_first_bone_group.unsqueeze(-1), 
                bone_local_oris, 
                torch.tensor([1, 0, 0, 0], device=self.device).view(1, 1, 4)
            ) # (1, bn, 4) quaternions (w, x, y, z), set to identity quat for non-optimized bones
            loss = kpt_repr_plus_bone_pose_and_length_loss(
                m_kpts,
                self.fish.bone_angle_priors,
                self.fish.bone_length_min,
                self.fish.bone_length_max,
                proj_m_from_blworld,
                kpts_2d,
                kpts_conf,
                recombined_body_pose.flatten(1),
                bone_local_oris_bone_group,
                recombined_body_bone_length,
                angle_constraint_weight=self.angle_constraint_weight,
                smooth_weight=self.smooth_weight,
                bone_length_constraint_weight=self.bone_length_constraint_weight,
            )
            # Silhouette loss
            silhouette_renders = silhouette_renderer(
                out["vertices"], self.faces.unsqueeze(0), global_t
            )
            loss += mask_fitting_loss(
                silhouette_renders, masks.float(), 0.1 * self.mask_weight
            )
            if not torch.isfinite(loss):
                raise ValueError("Error: Stage 2 produced non-finite loss. Stopping Stage 2 early.")
            opt_body.zero_grad()
            loss.backward()
            if has_nonfinite_grads([optimizable_body_pose, optimizable_body_bone_length, global_orient, global_t, scale]):
                print(f"Stage 2 - global ori grad: {global_orient.grad}")
                print(f"Stage 2 - global t grad: {global_t.grad}")
                print(f"Stage 2 - scale grad: {scale.grad}")
                print(f"Stage 2 - optimizable body pose grad: {optimizable_body_pose.grad}")
                print(f"Stage 2 - optimizable body bone length grad: {optimizable_body_bone_length.grad}")
                raise ValueError("Error: Stage 2 produced non-finite gradients. Stopping Stage 2 early.")
            opt_body.step()
        
        # synchronise body pose and body bone length to optimized state
        body_pose = recombine_frozen_and_optimized_tensor(frozen_body_pose, optimizable_body_pose, in_first_bone_group)
        body_bone_length = recombine_frozen_and_optimized_tensor(frozen_body_bone_length, optimizable_body_bone_length, in_first_bone_group)


        if len(self.fish.bone_groups) > 1:
            # ============ Stage 3: optimize remaining bone groups individually ============
            # optimize:
            # -----------------------------------
            # body_pose, 
            # bone_length, 
            # global_t,
            # global_orient 
            # scale
            # --------------
            # loss: 
            # -----------------------------------
            # (keypoint L2-distance * keypoint confidence²) of keypoints of that bone group, 
            # mask l1 distance,
            # bone constraint loss (angle/length min and max)
            # smooth loss compared to previous stage (-> L2 distance of bone orientation Rodriguez vectors and L2 distance of bone lengths) (! this loss is not present in stage 2 !)
            # ===================================

            for bg_idx, bone_group in enumerate(self.fish.bone_groups[1:]):
                # the loop starts enumerating at the second bone group, so bg_idx 0 is actually bg_idx 1 and so on
                bg_idx = bg_idx+1 
                # body pose excludes head joint so each bone index is actually one higher
                in_bone_group = torch.tensor(
                    [(b_idx + 1) in bone_group for b_idx in range(body_pose.size(1))],
                    dtype=torch.bool, device=self.device
                )
                kpt_group = self.fish.keypoint_groups[bg_idx]
                not_in_kpt_group = torch.tensor(
                    [k_idx not in kpt_group for k_idx in range(kpts_conf.size(1))],
                    dtype=torch.bool, device=self.device
                )

                # disable optimization on non-bone-group slices of body pose & bone length tensors
                with torch.no_grad():
                    frozen_body_pose = body_pose.clone().detach()
                    frozen_body_bone_length = body_bone_length.clone().detach()
                optimizable_body_pose        = body_pose[:, in_bone_group, :].detach().clone().requires_grad_(True)
                optimizable_body_bone_length = body_bone_length[:, in_bone_group].detach().clone().requires_grad_(True)

                opt_bone_group = torch.optim.Adam(
                    [optimizable_body_pose, optimizable_body_bone_length, global_orient, global_t, scale],
                    lr=self.step_size,
                )

                # reset kpts_conf because we manually set some entries to 0 in stage 2
                reset_kpts_conf()

                # Disable keypoints that don't belong to the bone group for this stage
                kpts_conf[:, not_in_kpt_group] = 0

                init_bp = body_pose.clone().detach()
                init_bl = body_bone_length.clone().detach()
                for i in range(self.num_iters):
                    # print(f"Stage 3 - global ori: {global_orient}")
                    # print(f"Stage 3 - body global t: {global_t}")
                    # print(f"Stage 3 - scale: {scale}")
                    # print(f"Stage 3 - body pose: {body_pose}")
                    # print(f"Stage 3 - body bone length: {body_bone_length}")
                    recombined_body_pose = recombine_frozen_and_optimized_tensor(frozen_body_pose, optimizable_body_pose, in_bone_group)
                    recombined_body_bone_length = recombine_frozen_and_optimized_tensor(frozen_body_bone_length, optimizable_body_bone_length, in_bone_group)
                    out = self.fish(
                        global_ori=global_orient,
                        body_pose=recombined_body_pose.flatten(1),
                        body_bone_length=recombined_body_bone_length,
                        scale=scale,
                    )
                    m_kpts = out["keypoints"].to(self.device) + global_t
                    m_kpts = m_kpts.expand(batch_size, -1, -1)
                    bone_local_oris = out["body_bone_poses_rest_bone_spaces"].to(self.device)
                    bone_local_oris_bone_group = torch.where(
                        in_bone_group.unsqueeze(-1), 
                        bone_local_oris, 
                        torch.tensor([1, 0, 0, 0], device=self.device).view(1, 1, 4)
                    ) # (1, bn, 4) quaternions (w, x, y, z), set to identity quat for non-optimized bones
                    loss = kpt_repr_plus_bone_pose_and_length_loss(
                        m_kpts,
                        self.fish.bone_angle_priors,
                        self.fish.bone_length_min,
                        self.fish.bone_length_max,
                        proj_m_from_blworld,
                        kpts_2d,
                        kpts_conf,
                        recombined_body_pose.flatten(1),
                        bone_local_oris_bone_group,
                        recombined_body_bone_length,
                        angle_constraint_weight=self.angle_constraint_weight,
                        smooth_weight=self.smooth_weight,
                        bone_length_constraint_weight=self.bone_length_constraint_weight,
                        pose_init=init_bp.flatten(1),
                        bone_init=init_bl,
                    )
                    # Silhouette loss
                    silhouette_renders = silhouette_renderer(
                        out["vertices"], self.faces.unsqueeze(0), global_t
                    )
                    loss += mask_fitting_loss(
                        silhouette_renders, masks.float(), 0.1 * self.mask_weight
                    )
                    if not torch.isfinite(loss):
                        raise ValueError(f"Error: Stage 3 bone group {bg_idx} produced non-finite loss. Stopping this group early.")
                    opt_bone_group.zero_grad()
                    loss.backward()
                    if has_nonfinite_grads([optimizable_body_pose, optimizable_body_bone_length, global_orient, global_t, scale]):
                        raise ValueError(f"Error: Stage 3 bone group {bg_idx} produced non-finite gradients. Stopping this group early.")
                    opt_bone_group.step()

                # synchronise body pose and body bone length to optimized state
                body_pose = recombine_frozen_and_optimized_tensor(frozen_body_pose, optimizable_body_pose, in_bone_group)
                body_bone_length = recombine_frozen_and_optimized_tensor(frozen_body_bone_length, optimizable_body_bone_length, in_bone_group)


        # Gather final outputs
        vertices = out["vertices"].detach().cpu()
        # Flatten body_pose back to (B, J*3) for the concatenation
        pose = torch.cat([global_orient, body_pose.reshape(body_pose.shape[0], -1)], dim=-1).detach().cpu()
        bone = body_bone_length.detach().cpu()
        scale = scale.detach().cpu()
        translation = global_t.detach().cpu()
        return vertices, pose, bone, scale, translation, (0, 0)
