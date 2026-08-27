from typing import Optional

import torch
import torch.nn.functional as F
from src import fish_model_edit as fish_model
from src.Silhouette_Renderer_edit import Silhouette_Renderer
from src.losses_edit import (
    kpt_reprojection_loss,
    bbox_reference_scales,
    bone_angle_constraint_loss,
    bone_length_constraint_loss,
    init_deviation_loss,
    mask_fitting_loss,
    decompose_to_swing_twist,
    soft_iou_loss
)


class OptimizeMV:
    """
    Perform multi-view mesh fitting in three stages:
      1. Global rotation, translation, scale
      2. Body-pose and bone-length refinement
      3. Tail and silhouette offset refinement

    On call, returns posed vertices, pose parameters, bone lengths,
    scale, translation, and a dict with final component losses.
    """

    def __init__(
        self,
        fish_model_obj: fish_model.fish_model,
        angle_constraint_weight: float = 1.0,
        smooth_weight: float = 1.0,
        big_artic_weight: float = 1.0,
        bone_length_constraint_weight: float = 1.0,
        mask_weight: float = 1.0,
        keypoints_weight: float = 1.0,
        view_weights: Optional[list[float]] = None,
        step_size: float = 1e-2,
        num_iters: int = 100,
        device=torch.device("cpu"),
        scene_scale: float = 1.0,
    ):
        """
        Args:
            step_size: base Adam step size. It is interpreted as a *relative* step and is turned
                into per-parameter-group learning rates in `_build_optimizer`, see the comment
                there.
            scene_scale: CLAUDE FIX. A characteristic length of the capture volume in calibration
                world units, normally `CameraGroup.scene_scale` (the mean camera baseline). Used to
                express the translation step size in the calibration's own units.
        """
        # Store hyper-parameters
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.angle_constraint_weight = angle_constraint_weight
        self.smooth_weight = smooth_weight
        self.big_artic_weight = big_artic_weight
        self.bone_length_constraint_weight = bone_length_constraint_weight
        self.mask_weight = mask_weight
        self.keypoints_weight = keypoints_weight
        self.view_weights = view_weights
        self.scene_scale = float(scene_scale)

        # Load parametric fish mesh and faces
        self.fish = fish_model_obj
        self.faces = self.fish.faces

        # Introduce constant factors for each loss to balance them because their magnitudes are different
        self.constant_factor_bone_angle_constraint_loss = 10.0
        self.constant_factor_bone_length_constraint_loss = 10.0
        self.constant_factor_scale_loss = 10.0
        self.constant_factor_init_deviation_loss = 2
        self.constant_factor_mask_loss = 1.5 # this is for the IoU loss
        # CLAUDE FIX: the silhouette term is the sum of an overlap term (soft IoU) and a smaller
        # area-normalized L1 term. The IoU part is scale invariant and gives a clean signal once the
        # rendered silhouette and the detected mask overlap, but its gradient vanishes when they do
        # not; the L1 part keeps a usable gradient in that regime so the fit can still recover. It
        # is deliberately weighted well below the IoU term so it shapes the basin of attraction
        # without dominating the fit once the two silhouettes are in contact.
        self.constant_factor_mask_l1_loss = 0.25
        self.constant_factor_kpt_loss = 0.0002

        # torch.autograd.set_detect_anomaly(True)

    # ------------------------------------------------------------------
    # CLAUDE FIX: optimizer construction is factored out so that a *single* Adam instance, with a
    # single set of moment estimates, is shared by every stage and every bone group of a frame.
    # Adam applies a bias correction that makes its first handful of steps much smaller than the
    # nominal learning rate. Building a fresh optimizer per stage meant paying that warm-up again
    # at the start of each stage, which with a budget of only `num_iters` steps per stage wastes a
    # noticeable share of the budget. Parameters are frozen and unfrozen per stage by masking their
    # gradients and restoring their values (see `_apply_active_masks` / `_restore_inactive`), which
    # reproduces the per-stage freezing semantics exactly while keeping the optimizer state alive.
    # ------------------------------------------------------------------
    def _build_optimizer(self, params: dict, init_scale: torch.Tensor) -> torch.optim.Adam:
        """
        Build the single per-frame Adam optimizer with per-parameter-group learning rates.

        CLAUDE FIX: the parameters being optimized live in completely different units -- rotations
        in radians, bone lengths as multipliers around 1, and a translation in whatever world units
        the camera calibration uses (which can easily be hundreds of units). Adam's per-step
        displacement is bounded by roughly the learning rate, so a single shared step size means the
        translation can only ever crawl: with a step size of 1e-2 and 100 iterations it could move
        at most ~1 world unit per stage, which is far less than a fish moves between frames in a
        typical calibration. Each group therefore gets a step size scaled into its own units:
        rotations and bone lengths keep the configured step size directly, the translation is scaled
        by the camera-rig baseline, and the scale parameter is scaled by its own current magnitude.
        """
        lr_rot = self.step_size
        lr_len = self.step_size
        lr_trans = self.step_size * max(self.scene_scale, 1e-6)
        lr_scale = self.step_size * max(abs(float(init_scale.reshape(-1)[0])), 1e-6)
        return torch.optim.Adam(
            [
                {"params": [params["global_orient"]], "lr": lr_rot},
                {"params": [params["body_pose"]], "lr": lr_rot},
                {"params": [params["body_bone_length"]], "lr": lr_len},
                {"params": [params["global_t"]], "lr": lr_trans},
                {"params": [params["scale"]], "lr": lr_scale},
            ]
        )

    @staticmethod
    def _apply_active_masks(params: dict, active: dict) -> None:
        """
        CLAUDE FIX: zero the gradients of every parameter entry that this stage does not optimize.

        This reproduces the previous behaviour, where non-active parameters were simply left out of
        the optimizer's parameter list, but without having to rebuild the optimizer.
        """
        for name, param in params.items():
            if param.grad is None:
                continue
            mask = active.get(name)
            if mask is None:
                param.grad.zero_()
            else:
                param.grad.mul_(mask)

    @staticmethod
    def _restore_inactive(params: dict, active: dict, snapshot: dict) -> None:
        """
        CLAUDE FIX: restore the values of parameter entries this stage does not optimize.

        Zeroing a gradient is not enough on its own: Adam keeps a momentum term, so a parameter with
        a zero gradient can still drift. Writing the pre-stage value back after every step keeps
        frozen bones exactly frozen, which is the semantics the staged schedule relies on.
        """
        with torch.no_grad():
            for name, param in params.items():
                mask = active.get(name)
                if mask is None:
                    param.copy_(snapshot[name])
                else:
                    param.copy_(torch.where(mask.bool(), param, snapshot[name]))

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
        seg_mask_present_mask: Optional[torch.Tensor] = None
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
            bboxes (vn, 4): detected bounding boxes (x0, y0, x1, y1) per view, used to make the
                keypoint error comparable between near and far views
            seg_mask_present_mask: (vn,) bool/float per view indicating which view has a valid detection
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

        # CLAUDE FIX: view weights are validated and built exactly once. There used to be a second,
        # unvalidated construction a few lines further down that silently overwrote this one, so
        # the checks for length, finiteness and non-negativity had no effect on the values actually
        # used.
        if self.view_weights is None:
            view_weights = torch.ones(batch_size, device=self.device, dtype=kpts_2d.dtype)
        else:
            if len(self.view_weights) != batch_size:
                raise ValueError(
                    f"view_weights length mismatch in OptimizeMV: got {len(self.view_weights)}, "
                    f"but current batch has {batch_size} views."
                )
            view_weights = torch.tensor(self.view_weights, device=self.device, dtype=kpts_2d.dtype)
            if not torch.isfinite(view_weights).all():
                raise ValueError("OptimizeMV received non-finite values in view_weights.")
            if (view_weights < 0).any():
                raise ValueError("OptimizeMV received negative values in view_weights; expected non-negative weights.")
        # in the extraction pipeline, undetected kpts have been set to conf=-1
        # in the loss function, conf is a coefficient of the loss -> clamp to 0, 1
        kpts_conf = kpts_conf.clamp(0.0, 1.0)

        # take care that views without detection (they have an all-zero dummy mask) don't contribute a mask loss:
        # mask-only weights: zero out views with no mask detection, WITHOUT touching
        # view_weights (which also drives the keypoint reprojection loss).
        if seg_mask_present_mask is not None:
            mask_present = torch.as_tensor(
                seg_mask_present_mask, device=self.device, dtype=view_weights.dtype
            )
            if mask_present.shape[0] != batch_size:
                raise ValueError(
                    f"seg_mask_present_mask length mismatch: got {mask_present.shape[0]}, "
                    f"but current batch has {batch_size} views."
                )
            mask_view_weights = view_weights * mask_present
        else:
            mask_view_weights = view_weights

        # CLAUDE FIX: per-view rescaling factors derived from the detected bounding boxes. They make
        # the keypoint reprojection error comparable across views: a view in which the animal
        # appears small produces small pixel errors and would otherwise be effectively ignored next
        # to a close-up view. Views without a usable box fall back to a factor of 1.
        bbox_scales = bbox_reference_scales(bboxes)
        if bbox_scales is not None:
            bbox_scales = bbox_scales.to(device=self.device, dtype=kpts_2d.dtype)

        # CLAUDE FIX: the detected masks are brought onto the renderer's rasterization grid once,
        # here, so that the silhouette comparison is valid for any configured render resolution.
        masks = silhouette_renderer.resize_masks_to_render_size(masks.float())

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

        final_losses: dict[str, Optional[float]] = {
            "kpt_reprojection_loss": None,
            "mask_fitting_loss": None,
            "bone_angle_constraint_loss": None,
            "bone_length_constraint_loss": None,
        }

        def _to_float(loss_tensor: torch.Tensor) -> float:
            return float(loss_tensor.detach().item())

        def silhouette_loss(silhouette_renders: torch.Tensor) -> torch.Tensor:
            """
            CLAUDE FIX: combined silhouette term -- soft IoU plus a low-weight, area-normalized L1
            difference between the detected mask and the rendered projection. See
            `constant_factor_mask_l1_loss` for why both are present.
            """
            iou_term = self.constant_factor_mask_loss * soft_iou_loss(
                silhouette_renders, masks, self.mask_weight, view_weights=mask_view_weights
            )
            l1_term = self.constant_factor_mask_l1_loss * mask_fitting_loss(
                silhouette_renders, masks, self.mask_weight, view_weights=mask_view_weights
            )
            return iou_term + l1_term

        # ===== Initialize parameters =====
        # CLAUDE FIX: all five parameter tensors are persistent leaves for the whole frame, kept at
        # full size. Stages restrict what they optimize through the `active` masks rather than by
        # slicing out fresh tensors, which is what lets one Adam instance (and its moment estimates)
        # survive across all stages.
        # global_orient: (1,3), body_pose: (1,B,3), bone_length: (1,B)
        global_orient     = init_ori_plus_pose[:, :3].detach().clone().requires_grad_(True)
        body_pose         = init_ori_plus_pose[:, 3:].detach().clone().view(1, self.fish.n_body_bones, 3).requires_grad_(True)
        body_bone_length  = init_body_bone_length.detach().clone().requires_grad_(True)
        global_t          = init_t.detach().clone().requires_grad_(True)
        scale             = init_scale.detach().clone().requires_grad_(True)

        params = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "body_bone_length": body_bone_length,
            "global_t": global_t,
            "scale": scale,
        }
        param_list = list(params.values())

        # ===== keep copies of initial body pose and bone length for smoothness prior =====
        init_global_ori       = global_orient.detach().clone()
        init_body_pose        = body_pose.detach().clone()
        init_body_bone_length = body_bone_length.detach().clone()
        init_scale            = init_scale.detach().clone()

        ones_like_pose  = torch.ones_like(body_pose)
        ones_like_bone  = torch.ones_like(body_bone_length)
        ones_like_ori   = torch.ones_like(global_orient)
        ones_like_t     = torch.ones_like(global_t)
        ones_like_scale = torch.ones_like(scale)

        opt = self._build_optimizer(params, init_scale)

        def snapshot_params() -> dict:
            return {name: param.detach().clone() for name, param in params.items()}

        def bone_group_masks(bone_group) -> tuple[torch.Tensor, torch.Tensor]:
            """(pose_mask (1,B,3), bone_length_mask (1,B)) for the given bone group."""
            # body pose excludes the head joint, so bone index b lives at body-pose index b-1
            in_group = torch.tensor(
                [(b_idx + 1) in bone_group for b_idx in range(body_pose.size(1))],
                dtype=torch.bool, device=self.device
            )
            return (
                in_group.view(1, -1, 1).expand_as(body_pose).to(body_pose.dtype),
                in_group.view(1, -1).expand_as(body_bone_length).to(body_bone_length.dtype),
            )

        def bone_angle_quats(out, bone_group=None) -> torch.Tensor:
            """
            CLAUDE FIX: assemble the body-bone rotations that the angle constraint is evaluated on.

            The model reports each bone's *own* rotation rather than the accumulated chain rotation,
            so the swing-twist limits mean what the template says they mean. Index 0 -- the head/root
            bone and global orientation -- is excluded from angle-prior checking entirely. Bones
            outside the stage's bone group are replaced by an identity rotation so that they
            contribute nothing.
            """
            bone_local_oris = out["global_ori_plus_body_pose_rest_bone_spaces"].to(self.device)
            identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=bone_local_oris.dtype).view(1, 1, 4)
            if bone_group is None:
                # global stage: no body-bone angle priors are active
                return identity.expand(bone_local_oris.shape[0], bone_local_oris.shape[1] - 1, 4)

            in_group = torch.tensor(
                [(b_idx + 1) in bone_group for b_idx in range(body_pose.size(1))],
                dtype=torch.bool, device=self.device
            )
            return torch.where(
                in_group.view(1, -1, 1),
                bone_local_oris[:, 1:],
                identity,
            )

        #******************************************************************************
        # ============ Stage 1: optimize global_orient, translation, scale ("7D-stage") ============
        # loss: 
        # -----------------------------------
        # keypoint reprojection (GMoF robustified, weighted by keypoint confidence squared,
        #   keypoints_weight, view_weights, and normalized by the per-view detection box size),
        # silhouette loss: soft IoU plus a low-weight area-normalized L1 difference between the
        #   detected mask and the rendered projection (weighted by mask_weight and view_weights),
        # no angle prior is applied to the global orientation in this stage,
        # smoothness: 
        #   global_t smoothness prior vs init_t (weighted by smooth_weight)
        #   global_orient smoothness prior vs init_global_orient (weighted by smooth_weight)
        #   scale smoothness prior vs init_scale (weighted by smooth_weight)
        # ===================================

        active_stage1 = {
            "global_orient": ones_like_ori,
            "body_pose": torch.zeros_like(body_pose),
            "body_bone_length": torch.zeros_like(body_bone_length),
            "global_t": ones_like_t,
            "scale": ones_like_scale,
        }
        snapshot_stage1 = snapshot_params()

        for _ in range(self.num_iters):

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
            kpt_loss = self.constant_factor_kpt_loss * kpt_reprojection_loss(
                model_kpts,
                proj_m_from_blworld,
                kpts_2d,
                kpts_conf,
                keypoints_weight=self.keypoints_weight,
                view_weights=view_weights,
                bbox_scales=bbox_scales,
            )
            # CLAUDE FIX: the global orientation is subject to the root bone's swing-twist priors in
            # this stage too, so the global stage can no longer settle on an orientation that the
            # template declares impossible and then hand it to stage 2 as a starting point.
            angle_loss = self.constant_factor_bone_angle_constraint_loss * bone_angle_constraint_loss(
                bone_angle_priors=self.fish.bone_angle_priors[:, 1:],
                global_ori_plus_body_pose_rest_head_spaces=bone_angle_quats(out, bone_group=None),
                angle_constraint_weight=self.angle_constraint_weight,
            )
            # smoothness loss vs initialization (previous frame)
            smoothness_loss = self.constant_factor_init_deviation_loss * init_deviation_loss(
                smooth_weight=self.smooth_weight,
                translation=global_t,
                translation_init=init_t,
                orientation=global_orient,
                orientation_init=init_global_ori,
                scale=scale,
                scale_init=init_scale,
                scale_weight=self.constant_factor_scale_loss
            )
            # Silhouette loss
            silhouette_renders = silhouette_renderer(
                out["vertices"], self.faces.unsqueeze(0), global_t
            )
            mask_loss = silhouette_loss(silhouette_renders)
            loss = kpt_loss + angle_loss + smoothness_loss + mask_loss
            final_losses["kpt_reprojection_loss"] = _to_float(kpt_loss)
            final_losses["mask_fitting_loss"] = _to_float(mask_loss)
            final_losses["bone_angle_constraint_loss"] = _to_float(angle_loss)
            if not torch.isfinite(loss):
                raise ValueError("Error: Stage 1 produced non-finite loss. Stopping Stage 1 early.")

            opt.zero_grad()
            loss.backward()
            if has_nonfinite_grads(param_list):
                raise ValueError("Error: Stage 1 produced non-finite gradients. Stopping Stage 1 early.")
            self._apply_active_masks(params, active_stage1)
            opt.step()
            self._restore_inactive(params, active_stage1, snapshot_stage1)

        #***********************************************************
        # ============ Stage 2: refine first bone group ("torso optimization") ============
        # optimize:
        # -----------------------------------
        # body_pose (first bone group only), 
        # bone_length (first bone group only), 
        # global_orient,
        # global_t, 
        # scale
        # --------------
        # loss: 
        # -----------------------------------
        # keypoint reprojection (GMoF robustified, weighted by keypoint confidence squared,
        #   keypoints_weight, view_weights, and normalized by the per-view detection box size)
        #   for keypoints of bone group 0,
        # silhouette loss: soft IoU plus a low-weight area-normalized L1 difference between the
        #   detected mask and the rendered projection (weighted by mask_weight and view_weights),
        # swing-twist angle constraint loss (weighted by angle_constraint_weight):
        #   - twist-limit violation (symmetric about the rest pose),
        #   - swing ellipse violation with locked-axis handling for zero swing priors,
        #   - excludes the head/root bone and global orientation from angle-prior checking,
        # bone-length min/max constraint loss (weighted by bone_length_constraint_weight),
        # smoothness:
        #   body pose smooth L1 and 
        #   body bone length L1 and
        #   global translation L1 and
        #   global orientation L1 and
        #   scale L1 loss compared to initialization (weighted by smooth_weight)
        # ===================================

        first_bone_group = self.fish.bone_groups[0]
        pose_mask_first, bone_mask_first = bone_group_masks(first_bone_group)
        first_kpt_group = self.fish.keypoint_groups[0]
        not_in_first_kpt_group = torch.tensor(
            [k_idx not in first_kpt_group for k_idx in range(kpts_conf.size(1))],
            dtype=torch.bool, device=self.device
        )

        active_stage2 = {
            "global_orient": ones_like_ori,
            "body_pose": pose_mask_first,
            "body_bone_length": bone_mask_first,
            "global_t": ones_like_t,
            "scale": ones_like_scale,
        }
        snapshot_stage2 = snapshot_params()

        # Disable keypoints for this step that don't belong to the first bone group
        kpts_conf[:, not_in_first_kpt_group] = 0

        for _ in range(self.num_iters):
            out = self.fish(
                global_ori=global_orient,
                body_pose=body_pose.flatten(1),
                body_bone_length=body_bone_length,
                scale=scale,
            )
            m_kpts = out["keypoints"].to(self.device) + global_t
            m_kpts = m_kpts.expand(batch_size, -1, -1)

            kpt_loss = self.constant_factor_kpt_loss * kpt_reprojection_loss(
                model_keypoints=m_kpts,
                proj_m=proj_m_from_blworld,
                keypoints_2d=kpts_2d,
                keypoints_conf=kpts_conf,
                keypoints_weight=self.keypoints_weight,
                view_weights=view_weights,
                bbox_scales=bbox_scales,
            )
            angle_loss = self.constant_factor_bone_angle_constraint_loss * bone_angle_constraint_loss(
                bone_angle_priors=self.fish.bone_angle_priors[:, 1:],
                global_ori_plus_body_pose_rest_head_spaces=bone_angle_quats(out, bone_group=first_bone_group),
                angle_constraint_weight=self.angle_constraint_weight,
            )
            bone_length_loss = self.constant_factor_bone_length_constraint_loss * bone_length_constraint_loss(
                bone_length=body_bone_length,
                bone_length_min=self.fish.bone_length_min,
                bone_length_max=self.fish.bone_length_max,
                bone_length_constraint_weight=self.bone_length_constraint_weight,
            )
            smoothness_loss = self.constant_factor_init_deviation_loss * init_deviation_loss(
                body_pose=body_pose,
                bone_length=body_bone_length,
                smooth_weight=self.smooth_weight,
                pose_init=init_body_pose,
                bone_init=init_body_bone_length,
                translation=global_t,
                translation_init=init_t,
                orientation=global_orient,
                orientation_init=init_global_ori,
                scale=scale,
                scale_init=init_scale,
                scale_weight=self.constant_factor_scale_loss
            )
            # Silhouette loss
            silhouette_renders = silhouette_renderer(
                out["vertices"], self.faces.unsqueeze(0), global_t
            )
            mask_loss = silhouette_loss(silhouette_renders)
            loss = kpt_loss + angle_loss + bone_length_loss + smoothness_loss + mask_loss
            final_losses["kpt_reprojection_loss"] = _to_float(kpt_loss)
            final_losses["mask_fitting_loss"] = _to_float(mask_loss)
            final_losses["bone_angle_constraint_loss"] = _to_float(angle_loss)
            final_losses["bone_length_constraint_loss"] = _to_float(bone_length_loss)

            if not torch.isfinite(loss):
                raise ValueError("Error: Stage 2 produced non-finite loss. Stopping Stage 2 early.")

            opt.zero_grad()
            loss.backward()
            if has_nonfinite_grads(param_list):
                raise ValueError("Error: Stage 2 produced non-finite gradients. Stopping Stage 2 early.")
            self._apply_active_masks(params, active_stage2)
            opt.step()
            self._restore_inactive(params, active_stage2, snapshot_stage2)

        if len(self.fish.bone_groups) > 1:
            #*******************************************************************************
            # ============ Stage 3: optimize remaining bone groups individually ("limb optimization") ============
            # optimize:
            # -----------------------------------
            # body_pose (this bone group only), 
            # bone_length (this bone group only), 
            # global_t,
            # global_orient (if bone group includes root bone)
            # --------------
            # NOT optimized here: scale. The overall size of the animal is a whole-body property
            # that stage 1 and stage 2 determine from all keypoints and the full silhouette; letting
            # an individual bone group adjust it later means a group that sees only a couple of
            # keypoints (e.g. the tail) can rescale the entire body to reduce its own local error.
            # --------------
            # loss: 
            # -----------------------------------
            # keypoint reprojection (GMoF robustified, weighted by keypoint confidence squared,
            #   keypoints_weight, view_weights, and normalized by the per-view detection box size)
            #   for keypoints of that bone group,
            # silhouette loss: soft IoU plus a low-weight area-normalized L1 difference between the
            #   detected mask and the rendered projection (weighted by mask_weight and view_weights),
            # swing-twist angle constraint loss (weighted by angle_constraint_weight):
            #   - twist-limit violation (symmetric about the rest pose),
            #   - swing ellipse violation with locked-axis handling for zero swing priors,
            #   - always includes the global orientation against the root bone's priors,
            # bone-length min/max constraint loss (weighted by bone_length_constraint_weight),
            # smoothness (!compared to previous stage 3 loop!):
            #   body pose and body bone L1 (weighted by big_artic_weight)
            #   global_orient L1 if root bone in this bone group (weighted by smooth_weight)
            #   global translation L1 (weighted by smooth_weight)
            #   (!! in stage 2, the smooth difference was determined by the difference to init parameters and weighted by smooth_weight !!)
            # ===================================

            for bg_idx, bone_group in enumerate(self.fish.bone_groups[1:]):
                # the loop starts enumerating at the second bone group, so bg_idx 0 is actually bg_idx 1 and so on
                bg_idx = bg_idx + 1
                pose_mask, bone_mask = bone_group_masks(bone_group)
                kpt_group = self.fish.keypoint_groups[bg_idx]
                not_in_kpt_group = torch.tensor(
                    [k_idx not in kpt_group for k_idx in range(kpts_conf.size(1))],
                    dtype=torch.bool, device=self.device
                )

                root_in_group = 0 in bone_group
                active_stage3 = {
                    "global_orient": ones_like_ori if root_in_group else torch.zeros_like(global_orient),
                    "body_pose": pose_mask,
                    "body_bone_length": bone_mask,
                    "global_t": ones_like_t,
                    # CLAUDE FIX: scale is frozen for the whole of stage 3, see the note above.
                    "scale": torch.zeros_like(scale),
                }
                snapshot_stage3 = snapshot_params()

                # reset kpts_conf because we manually set some entries to 0 in the previous stage
                reset_kpts_conf()

                # Disable keypoints that don't belong to the bone group for this stage
                kpts_conf[:, not_in_kpt_group] = 0

                # keep body pose and body bone length from previous stage for smoothness prior comparison, so that each bone group is optimized to be close to the previous stage's solution on the entire body pose and bone length
                prev_bp = body_pose.detach().clone()
                prev_bl = body_bone_length.detach().clone()
                global_orient_prev = global_orient.detach().clone()
                global_t_prev = global_t.detach().clone()

                for i in range(self.num_iters):
                    out = self.fish(
                        global_ori=global_orient,
                        body_pose=body_pose.flatten(1),
                        body_bone_length=body_bone_length,
                        scale=scale,
                    )
                    m_kpts = out["keypoints"].to(self.device) + global_t
                    m_kpts = m_kpts.expand(batch_size, -1, -1)

                    kpt_loss = self.constant_factor_kpt_loss * kpt_reprojection_loss(
                        model_keypoints=m_kpts,
                        proj_m=proj_m_from_blworld,
                        keypoints_2d=kpts_2d,
                        keypoints_conf=kpts_conf,
                        keypoints_weight=self.keypoints_weight,
                        view_weights=view_weights,
                        bbox_scales=bbox_scales,
                    )
                    angle_loss = self.constant_factor_bone_angle_constraint_loss * bone_angle_constraint_loss(
                        bone_angle_priors=self.fish.bone_angle_priors[:, 1:],
                        global_ori_plus_body_pose_rest_head_spaces=bone_angle_quats(out, bone_group=bone_group),
                        angle_constraint_weight=self.angle_constraint_weight,
                    )
                    bone_length_loss = self.constant_factor_bone_length_constraint_loss * bone_length_constraint_loss(
                        bone_length=body_bone_length,
                        bone_length_min=self.fish.bone_length_min,
                        bone_length_max=self.fish.bone_length_max,
                        bone_length_constraint_weight=self.bone_length_constraint_weight,
                    )
                    # smoothness loss vs previous stage's solution
                    smoothness_loss = init_deviation_loss(
                        body_pose=body_pose.flatten(1),
                        bone_length=body_bone_length,
                        smooth_weight=self.big_artic_weight,
                        pose_init=prev_bp.flatten(1),
                        bone_init=prev_bl,
                    ) # use big artic weight to allow articulation of bone groups to be more variable or more restricted, depending big artic weight
                    # other smoothness term summands still go with smooth_weight; these are not articulation-specific.
                    # CLAUDE FIX: no scale smoothness term here -- scale is frozen during stage 3, so
                    # a prior pulling it back towards its initialization would have no effect other
                    # than adding a constant to the reported loss.
                    smoothness_loss = smoothness_loss + self.smooth_weight * F.smooth_l1_loss(global_t, global_t_prev, reduction="sum")
                    if root_in_group:
                        smoothness_loss = smoothness_loss + self.smooth_weight * F.smooth_l1_loss(global_orient, global_orient_prev, reduction="sum")
                    smoothness_loss = self.constant_factor_init_deviation_loss * smoothness_loss
                    # Silhouette loss
                    silhouette_renders = silhouette_renderer(
                        out["vertices"], self.faces.unsqueeze(0), global_t
                    )
                    mask_loss = silhouette_loss(silhouette_renders)
                    loss = kpt_loss + angle_loss + bone_length_loss + smoothness_loss + mask_loss
                    final_losses["kpt_reprojection_loss"] = _to_float(kpt_loss)
                    final_losses["mask_fitting_loss"] = _to_float(mask_loss)
                    final_losses["bone_angle_constraint_loss"] = _to_float(angle_loss)
                    final_losses["bone_length_constraint_loss"] = _to_float(bone_length_loss)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Error: Stage 3 bone group {bg_idx} produced non-finite loss. Stopping this group early.")
                    opt.zero_grad()
                    loss.backward()
                    if has_nonfinite_grads(param_list):
                        raise ValueError(f"Error: Stage 3 bone group {bg_idx} produced non-finite gradients. Stopping this group early.")
                    self._apply_active_masks(params, active_stage3)
                    opt.step()
                    self._restore_inactive(params, active_stage3, snapshot_stage3)

        # ============ Final unmasked evaluation ============
        # `final_losses` must describe the mesh that is actually returned/rendered for this
        # frame. Two issues would otherwise make it misleading:
        #   1. Staleness: inside the stage loops, `final_losses` is captured from the forward
        #      pass BEFORE the loop's last `opt.step()`, i.e. one gradient step behind the
        #      parameters we actually return below.
        #   2. Partial coverage: during Stage 3, `kpts_conf` and the bone-angle quaternions are
        #      masked down to a single bone group per inner loop, so whatever values happen to
        #      survive from the very last bone group processed do NOT represent a whole-body
        #      loss (unlike bone_length_loss and mask_loss, which are always computed on the
        #      full body/silhouette already).
        # Recomputing once here, after every stage has finished, with the final parameters and
        # full (un-masked) keypoints/bones fixes both problems consistently for all four terms.
        with torch.no_grad():
            reset_kpts_conf()
            final_out = self.fish(
                global_ori=global_orient,
                body_pose=body_pose.flatten(1),
                body_bone_length=body_bone_length,
                scale=scale,
            )
            final_m_kpts = (final_out["keypoints"].to(self.device) + global_t).expand(batch_size, -1, -1)

            final_kpt_loss = self.constant_factor_kpt_loss * kpt_reprojection_loss(
                model_keypoints=final_m_kpts,
                proj_m=proj_m_from_blworld,
                keypoints_2d=kpts_2d,
                keypoints_conf=kpts_conf,
                keypoints_weight=self.keypoints_weight,
                view_weights=view_weights,
                bbox_scales=bbox_scales,
            )
            # CLAUDE FIX: the reported angle loss covers every bone, including the global
            # orientation, so it describes the whole returned pose rather than one bone group.
            final_bone_local_oris = final_out["global_ori_plus_body_pose_rest_bone_spaces"].to(self.device)
            final_angle_loss = self.constant_factor_bone_angle_constraint_loss * bone_angle_constraint_loss(
                bone_angle_priors=self.fish.bone_angle_priors[:, 1:],
                global_ori_plus_body_pose_rest_head_spaces=final_bone_local_oris[:, 1:],
                angle_constraint_weight=self.angle_constraint_weight,
            )
            final_bone_length_loss = self.constant_factor_bone_length_constraint_loss * bone_length_constraint_loss(
                bone_length=body_bone_length,
                bone_length_min=self.fish.bone_length_min,
                bone_length_max=self.fish.bone_length_max,
                bone_length_constraint_weight=self.bone_length_constraint_weight,
            )
            final_silhouette_renders = silhouette_renderer(
                final_out["vertices"], self.faces.unsqueeze(0), global_t
            )
            final_mask_loss = silhouette_loss(final_silhouette_renders)
            final_deviation_from_prev_frame = self.constant_factor_init_deviation_loss * init_deviation_loss(
                body_pose=body_pose,
                bone_length=body_bone_length,
                smooth_weight=self.smooth_weight,
                pose_init=init_body_pose,
                bone_init=init_body_bone_length,
                translation=global_t,
                translation_init=init_t,
                orientation=global_orient,
                orientation_init=init_global_ori,
                scale=scale,
                scale_init=init_scale,
                scale_weight=self.constant_factor_scale_loss
            )

        final_losses["kpt_reprojection_loss"] = _to_float(final_kpt_loss)
        final_losses["mask_fitting_loss"] = _to_float(final_mask_loss)
        final_losses["bone_angle_constraint_loss"] = _to_float(final_angle_loss)
        final_losses["bone_length_constraint_loss"] = _to_float(final_bone_length_loss)
        final_losses["final_deviation_from_prev_frame"] = _to_float(final_deviation_from_prev_frame)

        # Gather final outputs
        out = final_out
        vertices = out["vertices"].detach().cpu()
        # Flatten body_pose back to (B, J*3) for the concatenation
        pose = torch.cat([global_orient, body_pose.reshape(body_pose.shape[0], -1)], dim=-1).detach().cpu()
        bone = body_bone_length.detach().cpu()
        scale = scale.detach().cpu()
        translation = global_t.detach().cpu()
        return vertices, pose, bone, scale, translation, final_losses
