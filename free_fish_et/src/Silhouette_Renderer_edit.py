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
from src.constants_edit import *
from src.CameraGroups import CameraGroup, _camera_group_from_args


class Silhouette_Renderer:
    """
    A class which is used to define differentiable silhoutte renderers for multiple cameras.
    This class assumes that undistortion has been performed prior and that K is the intrinsic matrix which maps to the undistorted space.
    """
    def __init__(
        self,
        device: str,
        camera_group: CameraGroup,
    ):
        """Initialise a differentiable silhouette renderer for a batch of cameras."""
        self.device = torch.device(device)
        camera_group = _camera_group_from_args(camera_group).to(self.device)

        self.n_batches = camera_group.batch_size
        self.image_size = camera_group.image_size

        Rs = BLENDERWORLD_2_PYTORCH3D.to(device=self.device, dtype=camera_group.R.dtype) @ camera_group.R
        Rs = camera_group.R
        ts = camera_group.T @ BLENDERWORLD_2_PYTORCH3D.to(device=self.device, dtype=camera_group.R.dtype).transpose(0,1)
        principal_points = camera_group.principal_points
        focal_lengths = camera_group.focal_lengths_px

        focal_lengths = focal_lengths.to(self.device)
        principal_points = principal_points.to(self.device)


        blend_params = BlendParams(sigma=1e-2, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=(int(self.image_size[0]), int(self.image_size[1])),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        image_size_batch = self.image_size.unsqueeze(0).expand(self.n_batches, -1)
        cameras = PerspectiveCameras(
            image_size=image_size_batch,
            focal_length=focal_lengths,
            principal_point=principal_points,
            R=Rs,
            T=ts,
            in_ndc=False,
            device=self.device,
        )
        self.cameras = cameras

        self.silhouette_renderers = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )


    def __call__(self, vertices: torch.Tensor, faces: torch.Tensor, T: torch.Tensor):
        """
        Args:
            vertices (1, vn, 3): coordinates of vn different vertices in world space.
            faces (1, fn, 3): triangle faces with reference to the indices of the vertices from vertices that span them.
            T (3): uniform translation that is going to be applied to the vertices. (in DSKv2 context this is the global translation - the translation that translates the mesh from object to world space)
        """
        if not (vertices.size(2) == 3 and faces.size(2) == 3):
            raise ValueError("Silhouette_Renderer call: shape of input vertices or faces is not (N, v/f, 3)")
        
        if vertices.device != self.device:
            vertices = vertices.to(self.device)
        if faces.device != self.device:
            faces = faces.to(self.device)
        if T.device != self.device:
            T = T.to(self.device)

        vertices_translated = (vertices + T) #@ BLENDERWORLD_2_PYTORCH3D.to(device=self.device, dtype=vertices.dtype).transpose(0,1)

        fish_mesh = Meshes(verts=vertices_translated.repeat(self.n_batches, 1, 1), faces=faces.repeat(self.n_batches, 1, 1))

        # TODO: What does silhouettes look like?
        silhouettes = self.silhouette_renderers(
            meshes_world=fish_mesh.clone()
        )

        return silhouettes[..., 3] #* 1.99
