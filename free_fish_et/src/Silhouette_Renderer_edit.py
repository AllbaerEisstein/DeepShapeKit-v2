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
    Use FishEyeCameras for rendering if any of the cameras has any distortion parameter that is > 0.
    Else, use PerspectiveCameras.
    """
    def __init__(
        self,
        device: str,
        image_sizes: torch.Tensor,
        Ks: torch.Tensor,
        Rs: torch.Tensor,
        Ts: torch.Tensor,
        focals: torch.Tensor,
        distortions: torch.Tensor,
        deviation: torch.Tensor=torch.tensor([[0, 0, 0]]),
    ):
        """
        Args:
            image_sizes (cn, 2): rendered image pixel width and pixel height
            Ks (cn, 3, 3): intrinsic camera matrices for cn different cameras
            Rs (cn, 3, 3): rotation matrices in world coordinates for cn different cameras in the same reference frame
            Ts (cn, 3): translation from world reference frame to the camera position of cn different cameras
            focals (cn): focal lengths (mm) of cn different cameras
            principal_points (cn, 2): principal point offsets in pixel coordinates of cn different cameras
            distortions (cn, 5): distortion factors [rad1, rad2, tan1, tan2, rad3] of cn different cameras. Specify as all 0.0 if no distortion is present.
            deviation: 
        """
        assert all(image_sizes.shape[0] == param.shape[0] for param in [Ks, Rs, Ts, focals, distortions]), "each camera needs one each of the parameters image_size, K, R, T, focals, principal_point, distortion"
        # move to device
        self.device = torch.device(device)
        self.image_sizes = image_sizes.to(self.device)
        Ks = Ks.to(self.device)
        Rs = Rs.to(self.device)
        Ts = Ts.to(self.device)
        focals = focals.to(self.device)
        distortions = distortions.to(self.device)
        deviation = deviation.to(self.device)

        # NOTE: every camera projection will be be rasterized to the same image size because this is a requirement for batch-rendering
        max_img_width_height = (torch.max(image_sizes[:,0]), torch.max(image_sizes[:,1]))
        # update the principal points to fit the new image size
        # after rendering, the images will be cropped to original size again
        Ks[:,0,2] = (max_img_width_height[0]/2).unsqueeze(0).repeat(1,Ks.size(0))
        Ks[:,1,2] = (max_img_width_height[1]/2).unsqueeze(0).repeat(1,Ks.size(0))
        principal_points = torch.stack((Ks[:,0,2],Ks[:,1,2]),dim=1)
        blend_params = BlendParams(sigma=1e-2, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size = (int(max_img_width_height[0]), int(max_img_width_height[1])),
            blur_radius = 0.0,  # np.log(1./1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel = 1,
        )
        
        # TODO: possible to only set the cameras to fisheye where strictly necessary?
        if any([coeff > 0.0 for coeffs in distortions for coeff in coeffs]):
            cameras = FishEyeCameras(
                image_size = image_sizes,
                focal_length = focals,
                principal_point = principal_points,
                radial_params = torch.cat((distortions[:,:2], distortions[:,4])),
                tangential_params = distortions[:,2:4],
                use_radial = True,
                use_tangential = True,
                R=Rs,
                T=Ts,
                world_coordinates = True,
                use_thin_prism = False,
                device = device,
            )
            print("Silhouette_Renderer: using fish-eye cameras")
        else:
            # NOTE: Somehow, when the renderer is called, it expects K in the following shape:
            # K = [
            #     [fx,   0,   px,   0],
            #     [0,   fy,   py,   0],
            #     [0,    0,    0,   1],
            #     [0,    0,    1,   0],
            # ]
            cameras = PerspectiveCameras(
                image_size=image_sizes,
                focal_length=focals,
                principal_point=principal_points,
                R=Rs,
                T=Ts,
                in_ndc=False,
                device=self.device
            )
            print("Silhouette_renderer: using perspective cameras")
        self.cameras = cameras

        self.silhouette_renderers = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )


    def __call__(self, vertices: torch.Tensor, faces: torch.Tensor, T: torch.Tensor):
        """
        Args:
            vertices (vn, 3): coordinates of vn different vertices in world space.
            faces (fn, 3): triangle faces with reference to the indices of the vertices from vertices that span them.
            T (3): uniform translation that is going to be applied to the vertices.
        """
        assert (
            vertices.size(2) == 3 and faces.size(2) == 3
        ), "shape of vertices or faces is not (N, v/f, 3)"
        if vertices.device != self.device:
            vertices = vertices.to(self.device)
        if faces.device != self.device:
            faces = faces.to(self.device)
        if T.device != self.device:
            T = T.to(self.device)

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

        fish_mesh = Meshes(verts=vertices.repeat(self.image_sizes.size(0), 1, 1), faces=faces.repeat(self.image_sizes.size(0), 1, 1))

        # TODO: What does a silhouette look like? -> Crop to correct image size (if padding and cropping logic not yet implemented in dataloaders)
        silhouettes = self.silhouette_renderers(
            meshes_world=fish_mesh.clone()
        )

        return silhouettes[..., 3] * 1.99
