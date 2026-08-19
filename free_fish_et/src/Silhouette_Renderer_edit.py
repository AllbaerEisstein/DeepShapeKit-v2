
import torch

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


class Silhouette_Renderer:
    """Differentiable silhouette renderer for cameras described in Blender/CV space."""

    def __init__(self, device: str, camera_group: CameraGroup):
        self.device = torch.device(device)

        self.uniform_cg = camera_group.to(self.device)
        if self.uniform_cg.original_image_size_wh is None:
            raise ValueError("CameraGroup.image_size_wh is required for rendering silhouettes")
        
        # Use camera group with uniform image size for rendering
        if not self.uniform_cg.is_uniform_image_size():
            raise ValueError("Silhouette_Renderer requires a CameraGroup with uniform image size")

        dtype = self.uniform_cg.R.dtype
        self.n_batches = self.uniform_cg.batch_size

        R_custom_conv = self.uniform_cg.R
        t_custom_conv = self.uniform_cg.t
        CUSTOMCONV_2_P3D = self.uniform_cg.from_pytorch3d.transpose(1,2)
        R_p3d = torch.matmul(CUSTOMCONV_2_P3D, R_custom_conv)
        T_p3d = torch.matmul(CUSTOMCONV_2_P3D, t_custom_conv.unsqueeze(-1)).squeeze(-1)

        principal_points = self.uniform_cg.principal_points.to(self.device, dtype=dtype)
        focal_lengths = self.uniform_cg.focal_lengths_px.to(self.device, dtype=dtype)

        image_hw = self.uniform_cg.original_image_size_hw.to(self.device, dtype=dtype)

        height = int(torch.round(image_hw[0, 0]).item())
        width = int(torch.round(image_hw[0, 1]).item())

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=blur_radius,
            faces_per_pixel=50,
        )

        # I have no clue why we need this.....
        invert_xy = torch.tensor([
            [-1,0,0],
            [0,-1,0],
            [0,0,1]
        ])

        cameras = PerspectiveCameras(
            image_size=image_hw,
            focal_length=focal_lengths,
            principal_point=principal_points,
            R=invert_xy.to(device=self.device, dtype=dtype) @ R_p3d.transpose(1,2),
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
        #print(self.world_to_p3d_T)
        verts_p3d = torch.matmul(verts_blender, self.blworld_to_p3d_T)

        fish_mesh = Meshes(
            verts=verts_p3d.repeat(self.n_batches, 1, 1),
            faces=faces.repeat(self.n_batches, 1, 1),
        )

        silhouettes = self.silhouette_renderers(meshes_world=fish_mesh)
        return silhouettes[..., 3]
