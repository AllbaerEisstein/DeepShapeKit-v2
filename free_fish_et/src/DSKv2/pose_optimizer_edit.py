import torch
from src import fish_model
from src.losses import (
    camera_fitting_loss,
    body_fitting_loss,
    mask_fitting_loss
)

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
        lim_weight=1.0,
        prior_weight=1.0,
        bone_weight=1.0,
        mask_weight=1.0,
        smooth_weights=None,
        step_size=1e-2,
        num_iters=100,
        device=torch.device('cpu'),
        mesh='carp.json',
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
        self.fish = fish_model(device=device, mesh=mesh)
        self.faces = self.fish.faces.to(self.device)

    def __call__(
        self,
        init_pose,
        init_bone,
        init_t,
        init_scale,
        proj_m,
        keypoints,
        masks,
        silhouette_renderer,
        has_prev=False,
        img_filenames=None,
        index=None,
        bboxes=None
    ):
        """
        Args:
          init_pose: tensor (1, 3 + P) [global_orient + body_pose]
          init_bone: tensor (1, B) bone lengths
          init_t: tensor (1, 3) translation
          init_scale: tensor (1,) scale factor
          proj_m: tensor (V, 3, 4) projection matrices for V views
          keypoints: tensor (V, K, 3) 2D keypoints + confidence
          masks: tensor (V, H, W) silhouette masks
        """
        # ===== Prepare data =====
        batch_size = proj_m.shape[0]
        kpts_2d = keypoints[..., :2]
        kpts_conf = keypoints[..., 2].clone()
        # Disable unreliable keypoints
        kpts_conf[0, -3] = 0
        kpts_conf[1, -1] = 0

        # ===== Initialize parameters =====
        # global_orient: (1,3), body_pose: (1,P), bone_length: (1,B)
        global_orient = init_pose[:, :3].clone().detach()
        body_pose = init_pose[:, 3:].clone().detach()
        global_t = init_t.clone().detach()
        bone_length = init_bone.clone().detach()
        scale = init_scale.clone().detach()

        # Stage 1: optimize global_orient, translation, scale
        for param in [body_pose, bone_length]:
            param.requires_grad_(False)
        for param in [global_orient, global_t, scale]:
            param.requires_grad_(True)
        opt_global = torch.optim.Adam(
            [global_orient, global_t, scale], lr=self.step_size
        )
        for _ in range(self.num_iters):
            out = self.fish(global_pose=global_orient,
                            body_pose=body_pose,
                            bone_length=bone_length,
                            scale=scale)
            # Reprojection loss
            model_kpts = out['keypoints'] + global_t.unsqueeze(1)
            model_kpts = model_kpts.expand(batch_size, -1, -1)
            loss = camera_fitting_loss(
                model_kpts, proj_m, kpts_2d, kpts_conf) \
                + self.prior_weight * (global_t - init_t).abs().sum()
            # Silhouette loss
            sil_f = silhouette_renderer(out['vertices'], self.faces.unsqueeze(0), global_t, 'front')
            sil_b = silhouette_renderer(out['vertices'], self.faces.unsqueeze(0), global_t, 'bottom')
            loss += mask_fitting_loss(
                torch.cat([sil_f, sil_b], 0), masks.float(),
                0.1 * self.mask_weight)
            opt_global.zero_grad(); loss.backward(); opt_global.step()

        # Stage 2: refine body_pose, bone_length, global, scale
        for param in [body_pose, bone_length, global_orient, global_t, scale]:
            param.requires_grad_(True)
        opt_body = torch.optim.Adam(
            [body_pose, bone_length, global_orient, global_t, scale],
            lr=self.step_size
        )
        # relax tail keypoints
        kpts_conf = kpts_conf.fill_(0.8)
        kpts_conf[:, -3] = 0; kpts_conf[:, -1] = 0
        kpts_conf[keypoints[..., 2] == 0] = 0
        for _ in range(self.num_iters):
            out = self.fish(global_pose=global_orient,
                            body_pose=body_pose,
                            bone_length=bone_length,
                            scale=scale)
            m_kpts = out['keypoints'] + global_t.unsqueeze(1)
            m_kpts = m_kpts.expand(batch_size, -1, -1)
            loss = body_fitting_loss(
                m_kpts, proj_m, kpts_2d, kpts_conf,
                body_pose, bone_length,
                lim_weight=self.lim_weight,
                prior_weight=self.prior_weight,
                bone_weight=self.bone_weight
            )
            opt_body.zero_grad(); loss.backward(); opt_body.step()

        # Stage 3: tail + silhouette offset
        sil_offset = torch.zeros((2,3), device=self.device, requires_grad=True)
        for p in [body_pose, bone_length, global_orient, global_t, scale, sil_offset]:
            p.requires_grad_(True)
        opt_tail = torch.optim.Adam(
            [body_pose, bone_length, global_orient, global_t, scale, sil_offset],
            lr=self.step_size
        )
        # reweight tail points
        kpts_conf = kpts_conf.fill_(0.8)
        kpts_conf[0, -3] = 0.1; kpts_conf[0, -1] = 1
        kpts_conf[1, 0] = 1; kpts_conf[1, 2] = 1
        kpts_conf[keypoints[..., 2] == 0] = 0
        init_bp = body_pose.clone().detach()
        init_bl = bone_length.clone().detach()
        for _ in range(self.num_iters):
            out = self.fish(global_pose=global_orient,
                            body_pose=body_pose,
                            bone_length=bone_length,
                            scale=scale)
            m_kpts = out['keypoints'] + global_t.unsqueeze(1)
            m_kpts = m_kpts.expand(batch_size, -1, -1)
            loss = body_fitting_loss(
                m_kpts, proj_m, kpts_2d, kpts_conf,
                body_pose, bone_length,
                lim_weight=self.lim_weight,
                prior_weight=self.prior_weight,
                bone_weight=self.bone_weight,
                pose_init=init_bp,
                bone_init=init_bl
            )
            opt_tail.zero_grad(); loss.backward(); opt_tail.step()

        # Gather final outputs
        vertices = out['vertices'].detach().cpu()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach().cpu()
        bone = bone_length.detach().cpu()
        scale = scale.detach().cpu()
        translation = global_t.detach().cpu()
        return vertices, pose, bone, scale, translation, (0, 0)
