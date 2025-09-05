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


class Silhouette_Renderer:
    """
    A class which is used to define differentiable silhoutte renderers for multiple cameras.
    """
    def __init__(
        self,
        device: str,
        image_size: torch.Tensor,
        Ks: torch.Tensor,
        Rs: torch.Tensor,
        Ts: torch.Tensor,
        focals: torch.Tensor,
        principal_points: torch.Tensor,
        distortions: torch.Tensor,
        deviation: torch.Tensor=torch.tensor([[0, 0, 0]]),
    ):
        """
        Args:
            image_size (2): rendered image pixel width and pixel height
            Ks (cn, 3, 3): intrinsic camera matrices for cn different cameras
            Rs (cn, 3, 3): rotation matrices in world coordinates for cn different cameras in the same reference frame
            Ts (cn, 3): translation from world reference frame to the camera position of cn different cameras
            focals (cn): focal lengths (mm) of cn different cameras
            principal_points (cn, 2): principal point offsets in pixel coordinates of cn different cameras
            distortions (cn, 5): distortion factors [rad1, rad2, tan1, tan2, rad3] of cn different cameras. Specify as all 0.0 if no distortion is present.
            deviation: 
        """
        # move to device
        self.device = torch.device(device)
        self.image_size = image_size.to(self.device)
        Ks = Ks.to(self.device)
        Rs = Rs.to(self.device)
        Ts = Ts.to(self.device)
        focals = focals.to(self.device)
        principal_points = principal_points.to(self.device)
        distortions = distortions.to(self.device)
        deviation = deviation.to(self.device)

        blend_params = BlendParams(sigma=1e-2, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size= (int(image_size[0]), int(image_size[1])),
            blur_radius=0.0,  # np.log(1./1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=1,
        )
        ### TODO: Cameras can be batch. (Ncams, params(Ncams, ..., ))
        self.cameras = []
        self.silhouette_renderers = []
        cn = Ks.shape[0]
        for c in range(cn):
            if any(coeff != 0 for coeff in distortions[c]):
                camera = FishEyeCameras(
                    image_size=image_size.unsqueeze(0).repeat(cn,1),
                    focal_length=focals[c],
                    principal_point=principal_points[c],
                    radial_params=torch.cat((distortions[c][:2], distortions[c][4])),
                    tangential_params=distortions[c][2:4],
                    use_radial = True,
                    use_tangential = True,
                    R=Rs[c],
                    T=Ts[c],
                    world_coordinates = True,
                    use_thin_prism = False,
                    device = device,
                )
            else:
                camera = PerspectiveCameras(
                    image_size=image_size.unsqueeze(0).repeat(cn,1),
                    focal_length=focals[c],
                    principal_point=principal_points[c],
                    K=Ks[c],
                    R=Rs[c],
                    T=Ts[c],
                    in_ndc=False,
                    device=self.device
                )
            self.cameras.append(camera)

            self.silhouette_renderers.append(MeshRenderer(
                rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
                shader=SoftSilhouetteShader(blend_params=blend_params),
            ))


    def __call__(self, vertices: torch.Tensor, faces: torch.Tensor, T: torch.Tensor):
        """
        Args:
            vertices (vn, 3): coordinates of vn different vertices in world space.
            faces (fn, 3): triangle faces with reference to the indices of the vertices from vertices that span them.
            T (3): uniform translation that is going to be applied to the vertices.
            view_index (scalar): view index
        """
        assert (
            vertices.size(2) == 3 and faces.size(2) == 3
        ), "shape of vertices or faces is not (N, v/f, 3)"
        if vertices.device != faces.device:
            faces = faces.to(vertices.device)

        # put vertices into a box with edge length 2 at (0,0,0)
        # vertices = vertices[0] - torch.mean(vertices[0], 0) + t.to(self.device) #+ torch.tensor([-0.025,0,0], device=self.device)
        # max_val = max(torch.max(abs(vertices[:, 0])),
        #               torch.max(abs(vertices[:, 1])),
        #               torch.max(abs(vertices[:, 2])), )
        # # vertices = torch.cat([vertices, max_val * torch.ones([vertices.size(0), 1]).to(device)], dim=1).unsqueeze(0)
        # vertices = vertices.unsqueeze(0) / max_val + self.deviation
        #
        # fish_mesh = Meshes(verts=vertices,
        #                    faces=faces)
        #
        # silhouette = self.silhouette_renderer(meshes_world=fish_mesh.clone(), R=self.R, T=self.T)

        # vertices = vertices[0] - torch.mean(vertices[0], 0) + t
        # vertices = vertices.unsqueeze(0)
        vertices = vertices + T
        Rx_90 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).to(
            self.device
        )
        Ry_90 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]).to(
            self.device
        )
        Rz_90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).to(
            self.device
        )
        # vertices = torch.einsum('bij,bkj->bki', Rx_90, vertices)
        vertices = vertices @ Ry_90 @ Ry_90 @ Rz_90

        fish_mesh = Meshes(verts=vertices, faces=faces)

        silhouette_renders = []
        for silhouette_renderer in self.silhouette_renderers:
            silhouette = silhouette_renderer(
                meshes_world=fish_mesh.clone()
            )
            silhouette_renders.append(silhouette[..., 3] * 1.99)

        return torch.stack(silhouette_renders)
