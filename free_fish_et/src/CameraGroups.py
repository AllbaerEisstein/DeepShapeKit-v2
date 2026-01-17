
from dataclasses import dataclass
from typing import Optional, Union

import torch

from .geometry import perspective_projection
from .constants_edit import CV_2_BLENDERWORLD, PYTORCH3D_2_BLENDERWORLD


@dataclass
class CameraGroup:
    """Container for synchronized projection, intrinsic, and extrinsic parameters.
    Attributes:
        P: Projection matrices of shape (B, 3, 4)
        K: Intrinsic matrices of shape (B, 3, 3)
        R: Rotation matrices of shape (B, 3, 3)
        t: Translation vectors of shape (B, 3)
        from_blenderworld: Matrices to convert Blender-world coordinates to the camera's working coordinate system, shape (B, 3, 3)
        original_image_size_wh: Original image sizes (width, height) of shape (B, 2)
        target_uniform_image_size: If not None, the target uniform image size (width, height) for all cameras, shape (2,)
    """

    P: torch.Tensor
    K: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    from_blenderworld: torch.Tensor # A matrix M so that point3d_custom_coord_convention = M @ point3d_blenderworld
    original_image_size_wh: torch.Tensor
    target_uniform_image_size: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        for name in ("P", "K", "R", "t", "from_blenderworld", "original_image_size_wh"):
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"CameraGroup.{name} cannot be None")
            tensor = torch.as_tensor(value)
            if name in {"P", "K", "R"} and tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if name == "t" and tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if name == "image_size_wh":
                if tensor.ndim == 1:
                    tensor = tensor.expand(self.P.shape[0], -1)
                elif tensor.shape[0] == self.P.shape[0]:
                    tensor = tensor
                else:
                    raise ValueError("CameraGroup.original_image_size_wh must have shape (2,) or (batch_size, 2)")
            else:
                tensor = tensor.to(dtype=torch.float32)
            setattr(self, name, tensor.contiguous())

    @property
    def from_cv(self):
        cv_to_blworld = CV_2_BLENDERWORLD.to(self.from_blenderworld.device, dtype=self.from_blenderworld.dtype)
        cv_to_blworld = cv_to_blworld.unsqueeze(0).expand(self.batch_size, -1, -1)
        return torch.matmul(self.from_blenderworld, cv_to_blworld)

    @property
    def from_pytorch3d(self):
        p3d_to_blworld = PYTORCH3D_2_BLENDERWORLD.to(self.from_blenderworld.device, dtype=self.from_blenderworld.dtype)
        p3d_to_blworld = p3d_to_blworld.unsqueeze(0).expand(self.batch_size, -1, -1)
        return torch.matmul(self.from_blenderworld, p3d_to_blworld)

    @property
    def Rt(self) -> torch.Tensor:
        return torch.cat([self.R, self.t.view(self.t.size(0), 3, 1)], dim=2)

    @property
    def batch_size(self) -> int:
        return int(self.R.shape[0])

    @property
    def principal_points(self) -> torch.Tensor:
        return torch.stack((self.K[:, 0, 2], self.K[:, 1, 2]), dim=1)

    @property
    def focal_lengths_px(self) -> torch.Tensor:
        return torch.stack((self.K[:, 0, 0], self.K[:, 1, 1]), dim=1)

    @property
    def focal_scalar_px(self) -> torch.Tensor:
        return self.K[:, 0, 0]

    @property
    def camera_centers(self) -> torch.Tensor:
        t_column = self.t.unsqueeze(-1)
        centers = -torch.matmul(self.R.transpose(1, 2), t_column).squeeze(-1)
        return centers

    @property
    def original_image_size_hw(self) -> torch.Tensor:
        if self.original_image_size_wh is None:
            raise ValueError("CameraGroup.original_image_size_wh is required")
        return self.original_image_size_wh[..., [1, 0]]

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "CameraGroup":
        target_dtype = self.K.dtype if dtype is None else dtype
        image_size = None if self.original_image_size_wh is None else self.original_image_size_wh.to(device=device, dtype=target_dtype)
        return CameraGroup(
            P=self.P.to(device=device, dtype=target_dtype),
            K=self.K.to(device=device, dtype=target_dtype),
            R=self.R.to(device=device, dtype=target_dtype),
            t=self.t.to(device=device, dtype=target_dtype),
            from_blenderworld=self.from_blenderworld.to(device=device, dtype=target_dtype),
            original_image_size_wh=image_size,
        )
    
    def get_cg_for_new_image_size(self, target_size_wh: Optional[torch.Tensor] = None) -> "CameraGroup":
        """Return a CameraGroup with adjusted intrinsics for the target image size. (default: use target_uniform_image_size)
        Shift the principal point to a new location suitable for the new image size.
        This assumes that the original image was padded where left and right padding was equal, and top and bottom padding was equal.
        Args:
            target_size_wh: Optional torch.Tensor of shape (2,) or (batch_size, 2) specifying the target image size (width, height).
        """
        target_size_wh = self.target_uniform_image_size if target_size_wh is None else target_size_wh
        if target_size_wh is None:
            raise ValueError("target_size_wh argument or self.target_uniform_image_size is required to get adjusted CameraGroup")
        if self.original_image_size_wh is None:
            raise ValueError("CameraGroup.image_size_wh is required to get adjusted CameraGroup")
        
        if target_size_wh.ndim == 1:
            target_size_wh = target_size_wh.expand(self.batch_size, -1)
        elif target_size_wh.shape[0] != self.batch_size:
            raise ValueError("target_size_wh must have shape (2,) or (batch_size, 2)")
        
        w, h = self.original_image_size_wh[:,0], self.original_image_size_wh[:,1]
        target_w, target_h = target_size_wh[:,0], target_size_wh[:,1]

        pad_x = (target_w - w) / 2.0
        pad_y = (target_h - h) / 2.0

        K_new = self.K.clone().detach()
        K_new[:,0,2] += pad_x
        K_new[:,1,2] += pad_y
        P_new = K_new @ self.Rt

        return CameraGroup(
            P=P_new,
            K=K_new,
            R=self.R,
            t=self.t,
            from_blenderworld=self.from_blenderworld,
            original_image_size_wh=target_size_wh,
            target_uniform_image_size=None,
        ).to(device=self.K.device, dtype=self.K.dtype)
    
    def with_intrinsics_adjusted_for_uniform_image_size(self) -> "CameraGroup":
        """Return a CameraGroup with adjusted intrinsics for the target uniform image size."""
        return self.get_cg_for_new_image_size()
    
    def is_uniform_image_size(self) -> bool:
        """Return True if all cameras have the same image size."""
        if self.original_image_size_wh is None:
            return False
        first_size = self.original_image_size_wh[0]
        return torch.all(self.original_image_size_wh == first_size).item()

    def projection_matrices(self, blender: bool = False) -> torch.Tensor:
        """Return projection matrices. If ``blender`` is True they expect Blender-world points."""
        if not blender:
            return self.P
        R_blender = torch.matmul(self.R, self.from_blenderworld)
        Rt = torch.cat([R_blender, self.t.unsqueeze(-1)], dim=2)
        return torch.matmul(self.K, Rt)

    def _expand_points(self, points: torch.Tensor) -> tuple[torch.Tensor, bool]:
        squeeze = False
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze = True
        if points.shape[0] == 1:
            points = points.expand(self.batch_size, -1, -1)
        elif points.shape[0] != self.batch_size:
            raise ValueError("points batch dimension must be 1 or equal to number of cameras")
        return points, squeeze

    def points_from_blworld(self, points: torch.Tensor) -> torch.Tensor:
        """Convert Blender-world points to CV world coordinates."""
        points, squeeze = self._expand_points(points)
        F_T = self.from_blenderworld.transpose(1, 2)
        converted = torch.matmul(points, F_T)
        return converted.squeeze(0) if squeeze else converted

    def world_to_view_from_blworld(self, points: torch.Tensor) -> torch.Tensor:
        """Transform Blender-world points into camera coordinates."""
        pts_custom_conv = self.points_from_blworld(points)
        pts_custom_conv, squeeze = self._expand_points(pts_custom_conv)
        view = torch.einsum('bij,bkj->bki', self.R, pts_custom_conv) + self.t.unsqueeze(1)
        return view.squeeze(0) if squeeze else view

    def perspective_projection_from_blworld(self, points: torch.Tensor) -> torch.Tensor:
        """Project Blender-world points using the stored projection matrices."""
        pts_custom_conv = self.points_from_blworld(points)
        pts_custom_conv, squeeze = self._expand_points(pts_custom_conv)
        proj = perspective_projection(pts_custom_conv, self.K @ self.Rt)
        return proj.squeeze(0) if squeeze else proj

    def points_from_cv(self, points: torch.Tensor) -> torch.Tensor:
        """Convert CV world points to the camera's working coordinate system."""
        points, squeeze = self._expand_points(points)
        F_T = self.from_cv.transpose(1, 2)
        converted = torch.matmul(points, F_T)
        return converted.squeeze(0) if squeeze else converted

    def world_to_view_from_cv(self, points: torch.Tensor) -> torch.Tensor:
        """Transform CV world points into camera coordinates."""
        pts_custom = self.points_from_cv(points)
        pts_custom, squeeze = self._expand_points(pts_custom)
        view = torch.einsum('bij,bkj->bki', self.R, pts_custom) + self.t.unsqueeze(1)
        return view.squeeze(0) if squeeze else view

    def perspective_projection_from_cv(self, points: torch.Tensor) -> torch.Tensor:
        """Project CV world points using the stored projection matrices."""
        pts_custom = self.points_from_cv(points)
        pts_custom, squeeze = self._expand_points(pts_custom)
        proj = perspective_projection(pts_custom, self.P)
        return proj.squeeze(0) if squeeze else proj
