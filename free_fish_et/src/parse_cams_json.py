from typing import Any, Dict, List, Tuple
import numpy as np
import torch

KEY_ALIASES = {
    'P': ['P', 'projection', 'proj', 'projection_matrix', 'proj_m'],
    'K': ['K', 'intrinsic', 'intrinsics', 'camera_matrix'],
    'R': ['R', 'rotation', 'rot', 'rotation_matrix'],
    'T': ['T', 't', 'translation', 'translation_vector', 'translation_matrix', 'transform'],
    'f': ['f', 'focal_length', 'fx', 'fy', 'focal'],
    'distortion': ['distortion', 'distortions', 'dist'],
}

def _find_any(d: Dict[str, Any], aliases: List[str]):
    for k in aliases:
        if k in d:
            return d[k]
    lowered = {kk.lower(): kk for kk in d.keys()}
    for a in aliases:
        if a.lower() in lowered:
            return d[lowered[a.lower()]]
    return None

def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.array(x, dtype=float)
    return arr.astype(float)

def _ensure_shape(arr: np.ndarray, want_shape: Tuple[int, ...], transpose_if_needed: bool=False) -> np.ndarray:
    if arr is None:
        return None
    if arr.shape == want_shape:
        return arr
    if arr.ndim == 1 and np.prod(want_shape) == arr.size:
        return arr.reshape(want_shape)
    if transpose_if_needed and arr.shape == want_shape[::-1]:
        return arr.T
    if arr.ndim == 2:
        r, c = arr.shape
        wr, wc = want_shape
        if r >= wr and c >= wc:
            return arr[:wr, :wc]
    raise ValueError(f'Cannot coerce array shape {arr.shape} to desired {want_shape}')

def _parse_T(arr: Any) -> np.ndarray:
    a = _to_numpy(arr)
    if a is None:
        return None
    if a.ndim == 0:
        raise ValueError('Translation provided as scalar; expected length-3 or matrix containing translation.')
    if a.ndim == 1:
        if a.size == 3:
            return a.reshape(3,)
        if a.size == 4:
            if abs(a[3]) > 1e-8:
                return a[:3] / a[3]
            else:
                return a[:3]
        raise ValueError(f'1D translation array length {a.size} not supported (expect 3 or 4)')
    if a.ndim == 2:
        r, c = a.shape
        if (r == 3 and c == 1) or (r == 1 and c == 3):
            return a.reshape(3,)
        if (r == 3 and c >= 4):
            return a[:, 3].reshape(3,)
        if (r == 4 and c == 3):
            return a[3, :].reshape(3,)
        if (r == 4 and c == 4):
            t = a[:3, 3]
            return t.reshape(3,)
    raise ValueError(f'Could not parse translation from array with shape {a.shape}')

def _parse_R(arr: Any) -> np.ndarray:
    a = _to_numpy(arr)
    if a is None:
        return None
    if a.ndim == 2 and a.shape == (3,3):
        return a
    if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] >= 3:
        return a[:, :3]
    if a.ndim == 2 and a.shape == (4,4):
        return a[:3, :3]
    if a.ndim == 1 and a.size == 3:
        theta = np.linalg.norm(a)
        if theta < 1e-12:
            return np.eye(3)
        k = a / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
        return R
    if a.ndim == 1 and a.size == 4:
        q = a
        if abs(np.linalg.norm(q) - 1.0) > 1e-6:
            q = q / np.linalg.norm(q)
        w,x,y,z = (q[0], q[1], q[2], q[3])
        if abs(w) < 1e-6:
            w,x,y,z = q[3], q[0], q[1], q[2]
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        return R
    raise ValueError(f'Could not parse rotation matrix from array with shape {a.shape}')

def _parse_K(arr: Any) -> np.ndarray:
    a = _to_numpy(arr)
    if a is None:
        return None
    if a.ndim == 2 and a.shape == (3,3):
        return a
    if isinstance(arr, dict):
        fx = arr.get('fx') or arr.get('f') or arr.get('focal_length')
        fy = arr.get('fy') or fx
        cx = arr.get('cx') or 0.0
        cy = arr.get('cy') or 0.0
        K = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1.0]], dtype=float)
        return K
    if a.ndim == 1 and (a.size == 4 or a.size == 5):
        fx, fy, cx, cy = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        K = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1.0]], dtype=float)
        return K
    if a.ndim == 2 and a.shape[0] >= 3 and a.shape[1] >= 3:
        return a[:3, :3]
    raise ValueError(f'Could not parse intrinsics K from array with shape {a.shape}')

def _parse_focal(f_val: Any) -> Tuple[float,float]:
    if f_val is None:
        return None
    if isinstance(f_val, dict):
        fx = f_val.get('fx', f_val.get('f', None))
        fy = f_val.get('fy', f_val.get('f', None))
        if fx is None:
            raise ValueError('Focal dictionary missing fx/f')
        fy = fy if fy is not None else fx
        return float(fx), float(fy)
    a = _to_numpy(f_val)
    if a is None:
        return None
    if a.ndim == 0:
        return float(a), float(a)
    if a.ndim == 1:
        if a.size == 1:
            return float(a[0]), float(a[0])
        if a.size >= 2:
            return float(a[0]), float(a[1])
    raise ValueError(f'Could not parse focal from {f_val} (shape {a.shape})')

def _parse_distortion(d_obj: Any) -> Tuple[float,float,float,float,float]:
    if d_obj is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    a = _to_numpy(d_obj) if not isinstance(d_obj, dict) else None
    if a is not None and a.ndim == 1:
        if a.size == 4:
            return float(a[0]), float(a[1]), float(a[2]), float(a[3]), 0.0
        if a.size == 5:
            return float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])
    if isinstance(d_obj, dict):
        rad_1 = d_obj.get('rad_1') or d_obj.get('k1') or d_obj.get('k_1') or d_obj.get('k1_rad') or d_obj.get('k1_radial')
        rad_2 = d_obj.get('rad_2') or d_obj.get('k2') or d_obj.get('k_2')
        rad_3 = d_obj.get('rad_3') or d_obj.get('k3') or d_obj.get('k_3')
        tan_1 = d_obj.get('tan_1') or d_obj.get('p1')
        tan_2 = d_obj.get('tan_2') or d_obj.get('p2')
        rad_1 = float(rad_1) if rad_1 is not None else 0.0
        rad_2 = float(rad_2) if rad_2 is not None else 0.0
        rad_3 = float(rad_3) if rad_3 is not None else 0.0
        tan_1 = float(tan_1) if tan_1 is not None else 0.0
        tan_2 = float(tan_2) if tan_2 is not None else 0.0
        return (rad_1, rad_2, tan_1, tan_2, rad_3)
    raise ValueError(f'Could not parse distortion from {d_obj}')

class CameraSet:
    def __init__(self, index_json: Dict[str, Any], views: List[str]):
        self.index_json = index_json
        self.views = views
        self.cam_matrices = index_json.get('camera_matrices', {})
        self.P_list: list[torch.Tensor] = []
        self.f_list: list[Tuple[float,float]] = []
        self.K_list: list[torch.Tensor] = []
        self.R_list: list[torch.Tensor] = []
        self.T_list: list[torch.Tensor] = []
        self.distortions_list: list[Tuple[float,float,float,float,float]] = []

        for v in self.views:
            matrices_json = self.cam_matrices.get(v, None)
            if matrices_json is None:
                raise ValueError(f'No camera matrices found for view "{v}" in index.json')

            P_raw = _find_any(matrices_json, KEY_ALIASES['P'])
            K_raw = _find_any(matrices_json, KEY_ALIASES['K'])
            R_raw = _find_any(matrices_json, KEY_ALIASES['R'])
            T_raw = _find_any(matrices_json, KEY_ALIASES['T'])
            f_raw = _find_any(matrices_json, KEY_ALIASES['f'])
            d_raw = _find_any(matrices_json, KEY_ALIASES['distortion'])

            if P_raw is None:
                raise ValueError(f'No projection matrix found for view "{v}" in index.json (checked aliases).')
            P_np = _to_numpy(P_raw)
            try:
                if P_np.ndim == 1 and P_np.size == 12:
                    P_np = P_np.reshape((3,4))
                elif P_np.ndim == 2 and P_np.shape == (4,3):
                    P_np = P_np.T
                elif P_np.ndim == 2 and P_np.shape == (4,4):
                    P_np = P_np[:3, :4]
                else:
                    P_np = _ensure_shape(P_np, (3,4), transpose_if_needed=True)
            except Exception as e:
                raise ValueError(f'Projection matrix P for view "{v}" could not be coerced to (3,4): {e}')

            try:
                K_np = _parse_K(K_raw if K_raw is not None else matrices_json.get('K', None))
            except Exception as e:
                raise ValueError(f'Intrinsic K for view "{v}" could not be parsed: {e}')

            try:
                R_np = _parse_R(R_raw if R_raw is not None else matrices_json.get('R', None))
            except Exception as e:
                raise ValueError(f'Rotation R for view "{v}" could not be parsed: {e}')

            try:
                T_np = _parse_T(T_raw if T_raw is not None else matrices_json.get('T', None))
            except Exception as e:
                raise ValueError(f'Translation T for view "{v}" could not be parsed: {e}')

            try:
                fx_fy = _parse_focal(f_raw if f_raw is not None else matrices_json.get('f', None))
            except Exception as e:
                raise ValueError(f'Focal length for view "{v}" could not be parsed: {e}')

            try:
                dist5 = _parse_distortion(d_raw if d_raw is not None else matrices_json.get('distortion', None))
            except Exception as e:
                raise ValueError(f'Distortion for view "{v}" could not be parsed: {e}')

            self.P_list.append(torch.tensor(P_np, dtype=torch.float32))
            self.K_list.append(torch.tensor(K_np, dtype=torch.float32))
            self.R_list.append(torch.tensor(R_np, dtype=torch.float32))
            self.T_list.append(torch.tensor(T_np.reshape(3,), dtype=torch.float32))
            self.f_list.append((float(fx_fy[0]), float(fx_fy[1])))
            self.distortions_list.append(tuple(float(x) for x in dist5))

    def get_camera_matrices(self, keep_focal_scalar: bool = True, focal_tol: float = 1e-6):
        """
        Return camera matrices and distortion parameters for all views.

        Returns:
            P_stack: (N, 3, 4) torch.float32
            K_stack: (N, 3, 3) torch.float32
            R_stack: (N, 3, 3) torch.float32
            T_stack: (N, 3)    torch.float32  <-- non-homogeneous translations
            focals:  either
                    (N,)    torch.float32  if every fx == fy and keep_focal_scalar==True
                    (N,2)   torch.float32  otherwise (fx,fy) per view
            dists:   (N, 5)    torch.float32  <-- (rad_1, rad_2, tan_1, tan_2, rad_3)
        """
        P_stack = torch.stack(self.P_list, 0)
        K_stack = torch.stack(self.K_list, 0)
        R_stack = torch.stack(self.R_list, 0)
        T_stack = torch.stack(self.T_list, 0)        # (N,3)
        focals_2 = torch.tensor(self.f_list, dtype=torch.float32)  # (N,2)
        dists = torch.tensor(self.distortions_list, dtype=torch.float32)  # (N,5)

        if keep_focal_scalar and focals_2.shape[0] > 0:
            # check fx == fy for all views within tolerance
            if torch.allclose(focals_2[:,0], focals_2[:,1], atol=focal_tol, rtol=0):
                focals = focals_2[:, 0].clone()  # (N,)
                return P_stack, K_stack, R_stack, T_stack, focals, dists
        # otherwise return (N,2)
        return P_stack, K_stack, R_stack, T_stack, focals_2, dists
