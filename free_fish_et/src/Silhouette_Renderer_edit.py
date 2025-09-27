
import torch

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

        cg = camera_group.to(self.device)
        if cg.image_size_wh is None:
            raise ValueError("CameraGroup.image_size_wh is required for rendering silhouettes")

        dtype = cg.R.dtype
        self.n_batches = cg.batch_size

        blworld_to_p3d = BLENDERWORLD_2_PYTORCH3D.to(self.device, dtype=dtype)
        blworld_to_p3d_inv = blworld_to_p3d.transpose(0, 1)
        cv_to_p3d = CV_2_PYTORCH3D.to(self.device, dtype=dtype)

        blworld_to_p3d_batch = blworld_to_p3d.unsqueeze(0).expand(self.n_batches, -1, -1)
        blworld_to_p3d_inv_batch = blworld_to_p3d_inv.unsqueeze(0).expand(self.n_batches, -1, -1)
        cv_to_p3d_batch = cv_to_p3d.unsqueeze(0).expand(self.n_batches, -1, -1)

        R_cv = cg.R
        t_cv = cg.t

        # Rotate into PyTorch3D camera coordinates and Blender -> PyTorch3D world basis.
        R_p3d = torch.matmul(cv_to_p3d_batch, torch.matmul(R_cv, blworld_to_p3d_inv_batch))

        # Recover camera centres in Blender world, convert to PyTorch3D world, then back to T.
        t_cv_col = t_cv.unsqueeze(-1)
        centres_bl = -torch.matmul(R_cv.transpose(1, 2), t_cv_col)
        centres_p3d = torch.matmul(blworld_to_p3d_batch, centres_bl)
        T_p3d = -torch.matmul(R_p3d, centres_p3d).squeeze(-1)

        principal_points = cg.principal_points.to(self.device, dtype=dtype)
        focal_lengths = cg.focal_lengths_px.to(self.device, dtype=dtype)

        image_hw = cg.image_size_hw.to(self.device, dtype=dtype)
        if image_hw.ndim == 1:
            image_size_batch = image_hw.unsqueeze(0).expand(self.n_batches, -1)
        elif image_hw.shape[0] == self.n_batches:
            image_size_batch = image_hw
        else:
            image_size_batch = image_hw.expand(self.n_batches, -1)
        if self.n_batches > 1:
            first_size = image_size_batch[0:1]
            if not torch.allclose(image_size_batch, first_size, atol=1e-4, rtol=0.0):
                raise ValueError("All cameras must share the same image size for batched rendering.")

        height = int(torch.round(image_size_batch[0, 0]).item())
        width = int(torch.round(image_size_batch[0, 1]).item())

        blend_params = BlendParams(sigma=1e-2, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        cameras = PerspectiveCameras(
            image_size=image_size_batch,
            focal_length=focal_lengths,
            principal_point=principal_points,
            R=R_p3d,
            T=T_p3d,
            in_ndc=False,
            device=self.device,
        )

        self.blworld_to_p3d = blworld_to_p3d
        self.blworld_to_p3d_T = blworld_to_p3d.transpose(0, 1)
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
