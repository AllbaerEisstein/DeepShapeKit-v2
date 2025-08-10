import os
import sys
#sys.path.append('/home/jonathan/miniforge3/envs/blender/lib/python3.13/site-packages')

import bpy
import bmesh

import numpy as np



def point_cloud(name: str, points: np.ndarray):
    """
    Create and link a point cloud object in Blender from an Nx3 array.
    """
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    obj = bpy.data.objects.new(name, mesh)
    mesh.from_pydata(points.tolist(), [], [])
    bpy.context.collection.objects.link(obj)
    return obj


def fill_mesh(grid_dist: float = 0.1, output_npz: str = None):
    """
    For the active mesh object in Edit Mode:
      - sample a 3D grid inside its bounding box
      - test which grid points lie inside the mesh
      - save inside points to NPZ
    """
    # ensure mesh utility is importable
    import is_inside_mesh as mesh_util
    
    obj = bpy.context.active_object
    bm = bmesh.from_edit_mesh(obj.data)

    # extract face vertex coordinates
    faces = np.array([[v.co[:] for v in f.verts] for f in bm.faces])
    verts = np.array([v.co[:] for v in obj.data.vertices])

    # generate grid and test containment
    grid = mesh_util.get_grid(verts, grid_dist=grid_dist)
    mask = mesh_util.is_inside_turbo(faces, grid)
    inside_pts = grid[mask]

    print(f"Grid points: {grid.shape[0]}, inside: {inside_pts.shape[0]}")

    # save
    if output_npz:
        np.savez(output_npz, points_inside=inside_pts)
    return inside_pts


def load_inside(path: str) -> np.ndarray:
    """Load previously saved inside points from .npz."""
    data = np.load(path)
    return data['points_inside']


def link_inside_points(npz_path: str):
    """Load inner points and create a Blender point cloud object."""
    pts = load_inside(npz_path)
    pc_obj = point_cloud("InnerPoints", pts)
    return pc_obj


def save_to_json(path: str):
    """
    Export current mesh + armature weights & joints to JSON.
    Expects:
      - active mesh in Edit Mode
      - an Armature named 'Armature' with bones in known order
    """
    # collect joint positions
    bone_names = [
        'Bone', 'Bone.001', 'Bone.002', 'Bone.003', 'Bone.004',
        'Bone.005', 'Bone.006', #'Bone.007', 'Bone.008', 'Bone.009', 'Bone.010'
    ]
    arm = bpy.data.objects['Armature']
    joints = [list(arm.data.bones[n].head_local) for n in bone_names]
    joints.append(list(arm.data.bones[bone_names[-1]].tail_local))

    # mesh geometry
    obj = bpy.context.active_object
    bm = bmesh.from_edit_mesh(obj.data)
    faces = [[v.index for v in f.verts] for f in bm.faces]
    vertices = [list(v.co) for v in obj.data.vertices]

    # vertex-group weights
    n_verts = len(obj.data.vertices)
    n_groups = len(obj.vertex_groups)
    weights = []
    for vi in range(n_verts):
        vw = [0.0]*n_groups
        for gi, group in enumerate(obj.vertex_groups):
            try:
                vw[gi] = group.weight(vi)
            except RuntimeError:
                pass
        weights.append(vw)

    # skeleton tree
    kintree = [list(range(-1, len(joints)-1)), list(range(len(joints)))]

    # assemble JSON
    out = {
        'V': vertices,
        'F': faces,
        'J': joints,
        'kintree_table': kintree,
        'weights': weights
    }
    import json
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

PROJ_DIR = '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data'
save_to_json(os.path.join(PROJ_DIR, 'bluegill_mesh.json'))

# Example usage within Blender Python console:
# bpy.ops.object.mode_set(mode='EDIT')
# points = fill_mesh(grid_dist=0.2, output_npz=os.path.join(PROJ_DIR, 'fish_inner.npz'))
# link_inside_points(os.path.join(PROJ_DIR, 'fish_inner.npz'))
# save_to_json(os.path.join(PROJ_DIR, 'fish_export.json'))
