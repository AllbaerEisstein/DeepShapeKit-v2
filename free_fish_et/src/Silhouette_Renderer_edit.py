from typing import Optional

import torch
import torch.nn.functional as F

import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    SoftSilhouetteShader,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
)

from src.CameraGroups import CameraGroup
from src.constants_edit import BLENDERWORLD_2_PYTORCH3D, CV_2_PYTORCH3D
from src.geometry import perspective_projection


class Silhouette_Renderer:
    """Differentiable silhouette renderer for cameras described in Blender/CV space."""

    def __init__(
        self,
        device: str,
        camera_group: CameraGroup,
        sigma: float = 1e-4,
        gamma: float = 1e-4,
        render_scale: float = 1.0,
    ):
        """
        Args:
            render_scale: CLAUDE FIX. Linear resolution factor for the rasterization grid, relative
                to the calibrated image size. 1.0 renders silhouettes at full camera resolution;
                0.5 halves each side and therefore rasterizes a quarter of the pixels. Rasterizing
                a full-resolution grid with many faces per pixel, once per optimizer iteration and
                per view, dominates both runtime and GPU memory, and the silhouette term does not
                need full resolution to be informative. The camera intrinsics are scaled to match,
                so the rendered silhouette stays geometrically consistent with the calibration, and
                `resize_masks_to_render_size()` brings the detected masks onto the same grid.
        """
        self.device = torch.device(device)

        self.uniform_cg = camera_group.to(self.device)
        if self.uniform_cg.original_image_size_wh is None:
            raise ValueError("CameraGroup.image_size_wh is required for rendering silhouettes")

        # Use camera group with uniform image size for rendering
        if not self.uniform_cg.is_uniform_image_size():
            raise ValueError("Silhouette_Renderer requires a CameraGroup with uniform image size")

        dtype = self.uniform_cg.R.dtype
        self.n_batches = self.uniform_cg.batch_size

        if not (0.0 < float(render_scale) <= 1.0):
            raise ValueError(
                f"render_scale must be in (0, 1]; got {render_scale}. Values above 1 would "
                f"rasterize at a higher resolution than the cameras were calibrated for."
            )
        self.render_scale = float(render_scale)

        R_custom_conv = self.uniform_cg.R
        t_custom_conv = self.uniform_cg.t
        CUSTOMCONV_2_P3D = self.uniform_cg.from_pytorch3d.transpose(1, 2)
        R_p3d = torch.matmul(CUSTOMCONV_2_P3D, R_custom_conv)
        T_p3d = torch.matmul(CUSTOMCONV_2_P3D, t_custom_conv.unsqueeze(-1)).squeeze(-1)

        principal_points = self.uniform_cg.principal_points.to(self.device, dtype=dtype)
        focal_lengths = self.uniform_cg.focal_lengths_px.to(self.device, dtype=dtype)

        image_hw = self.uniform_cg.original_image_size_hw.to(self.device, dtype=dtype)

        full_height = int(torch.round(image_hw[0, 0]).item())
        full_width = int(torch.round(image_hw[0, 1]).item())

        # CLAUDE FIX: rasterize on a grid scaled by `render_scale`, and scale focal length and
        # principal point by exactly the same factor so the projection itself is unchanged -- only
        # the sampling density differs. At least one pixel per side is always kept.
        height = max(1, int(round(full_height * self.render_scale)))
        width = max(1, int(round(full_width * self.render_scale)))
        self.full_image_size_hw = (full_height, full_width)
        self.render_image_size_hw = (height, width)

        # focal length and principal point are stored in (x, y) == (width, height) order
        render_scale_xy = torch.tensor(
            [width / full_width, height / full_height], device=self.device, dtype=dtype
        )
        focal_lengths = focal_lengths * render_scale_xy.view(1, 2)
        principal_points = principal_points * render_scale_xy.view(1, 2)
        render_image_hw = torch.tensor(
            [[height, width]], device=self.device, dtype=dtype
        ).expand(self.n_batches, -1)

        blend_params = BlendParams(sigma=sigma, gamma=gamma)
        blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=blur_radius,
            faces_per_pixel=50,
        )

        # I have no clue why we need this.....
        # (see check_camera_consistency() below, which verifies that whatever this convention
        #  conversion ends up doing agrees with the projection matrices used by the keypoint loss)
        invert_xy = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

        cameras = PerspectiveCameras(
            image_size=render_image_hw,
            focal_length=focal_lengths,
            principal_point=principal_points,
            R=invert_xy.to(device=self.device, dtype=dtype) @ R_p3d.transpose(1, 2),
            T=T_p3d,
            in_ndc=False,
            device=self.device,
        )

        BLWORLD_2_P3D = BLENDERWORLD_2_PYTORCH3D.to(self.device, dtype=dtype)
        self.blworld_to_p3d = BLWORLD_2_P3D
        self.blworld_to_p3d_T = BLWORLD_2_P3D.transpose(0, 1)
        self.cameras = cameras
        self.silhouette_renderers = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )

    def resize_masks_to_render_size(self, masks: torch.Tensor) -> torch.Tensor:
        """
        CLAUDE FIX: bring detected segmentation masks onto the rasterization grid.

        The silhouette loss compares the rendered alpha map against the detected mask pixel by
        pixel, so both have to live on the same grid. When `render_scale` is 1.0 this is a no-op.
        Bilinear resampling is used deliberately: the detected masks carry sub-pixel boundary
        information that nearest-neighbour sampling would discard.

        Args:
            masks: (vn, H, W) tensor at full camera resolution.
        Returns:
            (vn, h, w) float tensor matching `self.render_image_size_hw`.
        """
        masks = masks.float()
        if tuple(masks.shape[-2:]) == self.render_image_size_hw:
            return masks
        resized = F.interpolate(
            masks.unsqueeze(1),
            size=self.render_image_size_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(1)

    def check_camera_consistency(
        self,
        probe_points_blworld: Optional[torch.Tensor] = None,
        tolerance_px: float = 1.0,
        raise_on_mismatch: bool = False,
    ) -> float:
        """
        CLAUDE FIX: cross-validate this renderer's camera model against the projection matrices that
        the keypoint reprojection loss uses.

        The silhouette term and the keypoint term reach the image plane through two entirely
        independent code paths: PyTorch3D's `PerspectiveCameras` here, and
        `CameraGroup.projection_matrices(blender=True)` in the loss. If those two ever disagree,
        both terms still look individually plausible while pulling the mesh towards different image
        positions, and the fit settles on a meaningless compromise between them. Nothing else in
        the pipeline surfaces that.

        This projects a spread of probe points through both paths and reports the largest
        disagreement, converted back to full-resolution pixels so the tolerance is independent of
        `render_scale`.

        Args:
            probe_points_blworld: (N, 3) points in Blender-world coordinates. Defaults to a spread
                of points around the centroid of the camera rig.
            tolerance_px: disagreement above this is reported as a mismatch.
            raise_on_mismatch: raise instead of only warning.
        Returns:
            Maximum disagreement, in full-resolution pixels.
        """
        with torch.no_grad():
            dtype = self.uniform_cg.R.dtype
            if probe_points_blworld is None:
                from_bl_inv = self.uniform_cg.from_blenderworld.transpose(1, 2)
                centers_bl = torch.matmul(
                    from_bl_inv, self.uniform_cg.camera_centers.unsqueeze(-1)
                ).squeeze(-1)
                center = centers_bl.mean(dim=0)
                extent = self.uniform_cg.scene_scale * 0.1
                offsets = torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                        [0.5, 0.5, 0.5], [-0.5, 0.5, -0.5],
                    ],
                    device=self.device,
                    dtype=dtype,
                )
                probe_points_blworld = center.view(1, 3) + offsets * extent
            probe_points_blworld = probe_points_blworld.to(self.device, dtype=dtype)
            n_points = int(probe_points_blworld.shape[0])

            # Path A: the projection matrices the keypoint loss uses.
            proj_m = self.uniform_cg.projection_matrices(blender=True)
            batched = probe_points_blworld.unsqueeze(0).expand(self.n_batches, -1, -1)
            projected_matrix = perspective_projection(batched, proj_m)  # (vn, N, 2)

            # Path B: this renderer's PyTorch3D cameras, rescaled back to full resolution.
            points_p3d = torch.matmul(probe_points_blworld, self.blworld_to_p3d_T)
            projected_p3d = self.cameras.transform_points_screen(
                points_p3d.unsqueeze(0).expand(self.n_batches, -1, -1)
            )[..., :2]
            inv_scale_xy = torch.tensor(
                [
                    self.full_image_size_hw[1] / self.render_image_size_hw[1],
                    self.full_image_size_hw[0] / self.render_image_size_hw[0],
                ],
                device=self.device,
                dtype=dtype,
            )
            projected_p3d = projected_p3d * inv_scale_xy.view(1, 1, 2)

            max_disagreement = float((projected_matrix - projected_p3d).abs().max())

        if max_disagreement > tolerance_px:
            message = (
                f"Renderer cameras and keypoint projection matrices disagree by up to "
                f"{max_disagreement:.3f} px over {n_points} probe points "
                f"(tolerance {tolerance_px} px). The silhouette loss and the keypoint loss are "
                f"therefore optimizing towards different image positions; check the camera "
                f"convention conversion in Silhouette_Renderer.__init__."
            )
            if raise_on_mismatch:
                raise ValueError(message)
            print(f"Warning: {message}")
        return max_disagreement

    def __call__(self, vertices: torch.Tensor, faces: torch.Tensor, T: torch.Tensor):
        """Render silhouettes for a batch of cameras."""
        if not (vertices.size(-1) == 3 and faces.size(-1) == 3):
            raise ValueError("Silhouette_Renderer call expects (..., 3) shaped vertices/faces")

        if vertices.device != self.device:
            vertices = vertices.to(self.device)
        if faces.device != self.device:
            faces = faces.to(self.device)
        if T.device != self.device:
            T = T.to(self.device)

        verts_blender = vertices + T
        verts_p3d = torch.matmul(verts_blender, self.blworld_to_p3d_T)

        fish_mesh = Meshes(
            verts=verts_p3d.repeat(self.n_batches, 1, 1),
            faces=faces.repeat(self.n_batches, 1, 1),
        )

        silhouettes = self.silhouette_renderers(meshes_world=fish_mesh)
        return silhouettes[..., 3]