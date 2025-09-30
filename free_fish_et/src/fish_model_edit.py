"""
load articulated fish model from json file
modified from bird model by Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""

import os
import json
import torch
from .LBS_edit import LBS

# from .LBS_origin import LBS


class fish_model:
    """
    Parametric fish model.
    !! The fish model is expected to use the same coordinate system conventions as the camera matrices !!
    !! Members will only be on the device after calling to_device(target_device) !!
    Args:
        faces (F, 3): faces of the fish mesh each consisting of the indices of three vertices spanning it
        J (1, J, 3): joint locations in the rest pose (note: unsqueezed)
        V (1, V, 3): vertices in the rest pose (note: unsqueezed)
        vert2kpt (K, V): keypoint regression matrix
        weights (V, J): skinning weights
        n_body_bones (1): number of bones in the fish mesh
        bone_angle_min (nb*3): lower angle limits for each of the nb bones
        bone_angle_max (nb*3): upper angle limits for each of the nb bones
        bone_length_min (nb): lower length limit for each of the nb bones
        bone_length_max (nb): upper length limit for each of the nb bones
    On call perform linear blend skinning and return new vertex and keypoint positions.
    """

    def __init__(self, mesh_json_path: str):
        """
        Args:
            mesh: path to json file containing fish model
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device("cpu")
        self.device_active = False

        with open(mesh_json_path, "r") as infile:
            dd = json.load(infile)

        self.dd = dd

        self.n_bones = dd["n_bones"]
        # the first bone is treated differently from all the other bones (first bone: global orientation fitting; body bones: pose fitting)
        # So n_body_bones excludes the first bone
        self.n_body_bones = dd["n_bones"] - 1

        # we always have one more joint than bone (the head)
        self.n_joints = len(dd["J"])

        # triangulate if input mesh has quad faces
        fish_faces = dd["F"]
        triangle_faces = []
        for face in fish_faces:
            if len(face) == 4:
                triangle_faces.append([face[0], face[1], face[2]])
                triangle_faces.append([face[2], face[3], face[0]])
            elif len(face) == 3:
                triangle_faces.append(face)
            else:
                raise Exception("face data incorrect: got vertices number not in [3,4]")
        self.faces = torch.tensor(triangle_faces)

        self.kintree_table = torch.tensor(dd["kintree_table"])[:, :]
        self.parent_indices = self.kintree_table[0][:]

        self.weights = torch.tensor(dd["weights"])
        self.vert2kpt = torch.tensor(dd["vert2kpt"])

        self.J = torch.tensor(dd["J"]).unsqueeze(0) # (1,J,3)
        self.V = torch.tensor(dd["V"]).unsqueeze(0) # (1,V,3)

        # local coords, relative to head
        self.V = self.V - self.J[0, 0]
        self.J = self.J - self.J[0, 0]

        # scaling, unit conversion
        self.V = self.V #* 0.01
        self.J = self.J #* 0.01

        self.LBS = LBS(self.J, self.parent_indices, self.weights)

        # TODO: let user choose which indices belong to which bodypart (best to define in synthetic data generator)
        # Body_pose angle limit
        # angle limits are specified in exponential map (axis-angle where angle is specified as the length of the axis vector)
        # we minus index by 1 because we exclude root pose as it is modeled as global orient
        self.bone_angle_min = [0.0] * (self.n_body_bones * 3)
        self.bone_angle_max = [0.0] * (self.n_body_bones * 3)

        # repeat the pattern 0.0, -0.05, 0.0 for every bone (every three entries in bone_angle_min/max)
        for i in range(self.n_body_bones):
            self.bone_angle_min[i * 3 : (i + 1) * 3] = 0.0, -1.0, 0.0
            self.bone_angle_max[i * 3 : (i + 1) * 3] = -0.0, 1.0, 0.0

        # the last 3 bones should have different limits
        for i in range(self.n_body_bones - 3, self.n_body_bones):
            self.bone_angle_min[i * 3 : (i + 1) * 3] = 0.0, -1.0, 0.0
            self.bone_angle_max[i * 3 : (i + 1) * 3] = -0.0, 1.0, 0.0

        # the last bone should have a different limit
        for i in range(self.n_body_bones - 1, self.n_body_bones):
            self.bone_angle_min[i * 3 : (i + 1) * 3] = 0.0, -1.0, 0.0
            self.bone_angle_max[i * 3 : (i + 1) * 3] = -0.0, 1.0, 0.0

        # Body bone length limit
        self.bone_length_min = [1.0] * (self.n_body_bones)
        self.bone_length_max = [2.3] * (self.n_body_bones)

        self.bone_angle_max = torch.tensor(self.bone_angle_max)
        self.bone_angle_min = torch.tensor(self.bone_angle_min)
        self.bone_length_min = torch.tensor(self.bone_length_min)
        self.bone_length_max = torch.tensor(self.bone_length_max)


    def to_device(self, device: str):
        """
        Sends this fish_model instance to the specified device.
        Returns:
            None
        """
        self.device = torch.device(device)
        self.kintree_table = self.kintree_table.to(device)
        self.parent_indices = self.parent_indices.to(device)
        self.faces = self.faces.to(self.device)
        self.weights = self.weights.to(self.device)
        self.vert2kpt = self.vert2kpt.to(self.device)
        self.J = self.J.to(self.device)
        self.V = self.V.to(self.device)
        self.bone_angle_min = self.bone_angle_min.to(self.device)
        self.bone_angle_max = self.bone_angle_max.to(self.device)
        self.bone_length_min = self.bone_length_min.to(self.device)
        self.bone_length_max = self.bone_length_max.to(self.device)
        if not all(
            self.faces.device == attr.device
            for attr in [
                self.kintree_table,
                self.parent_indices,
                self.weights,
                self.vert2kpt,
                self.J,
                self.V,
                self.bone_angle_min,
                self.bone_angle_max,
                self.bone_length_min,
                self.bone_length_max,
            ]
        ):
            raise ValueError(
                "failed to move all fish object attributes to the same device"
            )
        self.LBS = LBS(self.J, self.parent_indices, self.weights)
        self.device = self.faces.device
        self.device_active = True


    def __call__(self, global_ori, body_pose, body_bone_length, scale=torch.tensor(1), pose2rot=True, deform=True):
        """
        Args:
            global_ori (BS, 3): BS (batch-size) different exponential map representations of global rotation
            body_pose (BS, bbn*3): exponential map representation of body pose (exclude root joint orient) -> orientation of bbn body bones
            body_bone_length (BS, bn): bone lengths for bbn body bones
            scale (BS, 1): scale factor
            pose2rot: if True, convert exponential map to rotation matrix inside LBS
            deform (bool): toggle if linear blend skinning should be applied. If not, just return rest-pose vertices and keypoints in local (model) space.
        Returns:
            keypoints (BS, kn, 3): coordinates of the kn keypoints (unsqueezed(0)) after LBS in local (model) space
            vertices (BS, vn, 3): coordinates of the vn vertices (unsqueezed(0)) after LBS in local (model) space
        """
        assert body_bone_length.size(1) == self.n_body_bones, f"body_bone_length must have size (batch_size, n_bones_total-1) but has size {body_bone_length.size()} instead of ({global_ori.size(0), self.n_body_bones})"
        assert global_ori.size(1) == 3, f"global_ori must have size (batch_size, 3) but has size {global_ori.size()} instead of ({global_ori.size(0), 3}) and must supply world space orientation in exponential map representation"
        assert body_pose.size(1) == self.n_body_bones*3, f"body_pose must have size (batch_size, (n_bones_total-1)*3) but has size {body_pose.size()} instead of ({global_ori.size(0), self.n_body_bones*3})"
        
        if not all(
            self.device == attr.device
            for attr in [self.faces, global_ori, body_pose, body_bone_length]
        ):
            global_ori = global_ori.to(self.device)
            body_pose = body_pose.to(self.device)
            body_bone_length = body_bone_length.to(self.device)


        batch_size = global_ori.shape[0]
        V = self.V.repeat([batch_size, 1, 1]) * scale

        # print(f"body_bone_length: {body_bone_length.size()}")
        # print(f"body_pose: {body_pose.size()}")

        # no need for global pose and body pose to be separate anymore
        # -> insert one length at the front (first bone) of the bone length tensor of each batch
        all_bone_lengths = torch.cat(
            [torch.ones([batch_size, 1], device=self.device), body_bone_length], dim=1
        )
        # concatenate global pose and body pose
        global_ori_plus_body_pose = torch.cat([global_ori, body_pose], dim=1)

        # print(f"all_bone_lengths: {all_bone_lengths.size()}")
        # print(f"global_ori_plus_body_pose: {global_ori_plus_body_pose.size()}")


        # LBS
        if deform:
            verts = self.LBS(V, global_ori_plus_body_pose, all_bone_lengths, scale, to_rotmats=pose2rot)
        else:
            verts = V

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = []
        for i in range(verts.shape[0]):
            kpt = torch.matmul(self.vert2kpt, verts[i])
            keypoints.append(kpt)
        keypoints = torch.stack(keypoints)

        # Final output after articulation
        # TODO: Why send to cpu? -> return a tuple and leave on device
        output = {"vertices": verts.cpu(), "keypoints": keypoints.cpu()}

        return output
