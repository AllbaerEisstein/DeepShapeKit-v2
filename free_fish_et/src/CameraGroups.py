
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class CameraGroup:
    """Container for synchronized projection, intrinsic, and extrinsic parameters."""

    P: torch.Tensor
    K: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    image_size_wh: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        for name in ("P", "K", "R", "t", "image_size_wh"):
            value = getattr(self, name)
            if value is None:
                continue
            tensor = torch.as_tensor(value)
            if name in {"P", "K", "R"} and tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if name == "t" and tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if name == "image_size_wh":
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.to(dtype=torch.float32)
            else:
                tensor = tensor.to(dtype=torch.float32)
            setattr(self, name, tensor.contiguous())

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
    def image_size_hw(self) -> torch.Tensor:
        if self.image_size_wh is None:
            raise ValueError("CameraGroup.image_size_wh is required for rendering")
        return self.image_size_wh[..., [1, 0]]

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "CameraGroup":
        target_dtype = self.K.dtype if dtype is None else dtype
        image_size = None if self.image_size_wh is None else self.image_size_wh.to(device=device, dtype=target_dtype)
        return CameraGroup(
            P=self.P.to(device=device, dtype=target_dtype),
            K=self.K.to(device=device, dtype=target_dtype),
            R=self.R.to(device=device, dtype=target_dtype),
            t=self.t.to(device=device, dtype=target_dtype),
            image_size_wh=image_size,
        )
