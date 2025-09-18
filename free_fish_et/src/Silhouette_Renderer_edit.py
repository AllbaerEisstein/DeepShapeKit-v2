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
from pytorch3d.renderer.fisheyecameras import FishEyeCameras

from src.constants_edit import CV_2_PYTORCH3D


class Silhouette_Renderer:
    """
    A class which is used to define differentiable silhoutte renderers for multiple cameras.
    This class assumes that undistortion has been performed prior and that K is the intrinsic matrix which maps to the undistorted space.
    """
    def __init__(
        self,
        device: str,
        image_size: torch.Tensor,
        Ks: torch.Tensor,
        Rs: torch.Tensor,
        ts: torch.Tensor,
        focals: torch.Tensor,
    ):
        """
        Args:
            image_size (2): rendered image pixel width and pixel height (expected to be uniform for all views)
            Ks (cn, 3, 3): intrinsic camera matrices for cn different cameras
            Rs (cn, 3, 3): rotation matrices in world coordinates for cn different cameras in the same reference frame
            ts (cn, 3): translation from world reference frame to the camera position of cn different cameras
            focals (cn): focal lengths (mm) of cn different cameras
        """
        assert all(Ks.shape[0] == param.shape[0] for param in [Rs, ts, focals]), "each camera needs one each of the parameters K, R, T, focal"
        self.n_batches = Rs.size(0)
        # move to device
        self.device = torch.device(device)
        Ks = Ks.to(self.device)
        Rs = Rs.to(self.device)
        ts = ts.to(self.device)
        focals = focals.to(self.device)
        self.image_size = image_size.to(device=self.device)

        # Transform to PyTorch3D-conventions (x left, y up, z forward)
        # --- input R and t are expected to transform from Blender world (x right, y forward, z up) to CV (x right, y down, z forward).
        # --- that means, a transformation from CV to PyTorch3D has to be appended.
        # --- input K is expected to transform from CV camera (x right, y down, z forward) to CV image (x right, y down, z forward).
        # --- that means, a transformation from CV to PyTorch3D has to be prepended.
        Rs = CV_2_PYTORCH3D.to(device=self.device, dtype=Rs.dtype).unsqueeze(0).expand(self.n_batches,-1,-1) @ Rs
        ts = CV_2_PYTORCH3D.to(device=self.device, dtype=ts.dtype) @ ts
        Ks = Ks @ torch.linalg.inv(CV_2_PYTORCH3D.to(device=self.device, dtype=Ks.dtype)).unsqueeze(0).expand(self.n_batches,-1,-1)
        # adjust principle points since the axes were flipped
        Ks[:,0,2] = self.image_size[0].unsqueeze(0).expand(self.n_batches) - Ks[:,0,2]
        Ks[:,1,2] = self.image_size[1].unsqueeze(0).expand(self.n_batches) - Ks[:,1,2]

        # NOTE: every camera projection will be be rasterized to the same image size because this is a requirement for batch-rendering
        principal_points = torch.stack((Ks[:,0,2],Ks[:,1,2]),dim=1)
        blend_params = BlendParams(sigma=1e-2, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size = (int(image_size[0]), int(image_size[1])),
            blur_radius = 0.0,  # np.log(1./1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel = 1,
        )
        
        # NOTE: Images are expected to be undistorted already
        ## Legacy-code:
        # if any([coeff > 0.0 for coeffs in distortions for coeff in coeffs]):
        #     cameras = FishEyeCameras(
        #         image_size = image_size,
        #         focal_length = focals,
        #         principal_point = principal_points,
        #         radial_params = torch.cat((distortions[:,:2], distortions[:,4])),
        #         tangential_params = distortions[:,2:4],
        #         use_radial = True,
        #         use_tangential = True,
        #         R=Rs,
        #         T=Ts,
        #         world_coordinates = True,
        #         use_thin_prism = False,
        #         device = device,
        #     )
        #     print("Silhouette_Renderer: using fish-eye cameras")
        # else:

        # NOTE: Somehow, when the renderer is called, it expects K in the following shape:
        # K = [
        #     [fx,   0,   px,   0],
        #     [0,   fy,   py,   0],
        #     [0,    0,    0,   1],
        #     [0,    0,    1,   0],
        # ]
        cameras = PerspectiveCameras(
            image_size=image_size.unsqueeze(0),
            focal_length=focals,
            principal_point=principal_points,
            R=Rs,
            T=ts,
            in_ndc=False,
            device=self.device
        )
        # print("Silhouette_renderer: using perspective cameras")
        self.cameras = cameras

        self.silhouette_renderers = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
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

        vertices = vertices + T

        fish_mesh = Meshes(verts=vertices.repeat(self.n_batches, 1, 1), faces=faces.repeat(self.n_batches, 1, 1))

        # TODO: What does silhouettes look like?
        silhouettes = self.silhouette_renderers(
            meshes_world=fish_mesh.clone()
        )

        return silhouettes[..., 3] #* 1.99
