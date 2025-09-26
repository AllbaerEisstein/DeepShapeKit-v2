
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class CameraGroup:
    """Container for batched projection, intrinsic, and extrinsic parameters."""

    P: torch.Tensor
    K: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    image_size: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        for name in ("P", "K", "R", "T", "image_size"):
            value = getattr(self, name)
            if value is None:
                continue
            if not torch.is_tensor(value):
                setattr(self, name, torch.as_tensor(value))

    @property
    def batch_size(self) -> int:
        return self.K.shape[0]

    @property
    def principal_points(self) -> torch.Tensor:
        return torch.stack((self.K[:, 0, 2], self.K[:, 1, 2]), dim=1)

    @property
    def focal_lengths_px(self) -> torch.Tensor:
        return torch.stack((self.K[:, 0, 0], self.K[:, 1, 1]), dim=1)

    @property
    def focal_scalar_px(self) -> torch.Tensor:
        return self.K[:, 0, 0]

    def to(self, device: Union[str, torch.device]) -> "CameraGroup":
        image_size = None if self.image_size is None else self.image_size.to(device)
        return CameraGroup(
            P=self.P.to(device),
            K=self.K.to(device),
            R=self.R.to(device),
            T=self.T.to(device),
            image_size=image_size,
        )


def _camera_group_from_args(cameras: CameraGroup) -> CameraGroup:
    """Validate that the provided object is a :class:`CameraGroup`."""

    if isinstance(cameras, CameraGroup):
        return cameras
    raise TypeError("Expected camera parameters to be provided as a CameraGroup instance.")
