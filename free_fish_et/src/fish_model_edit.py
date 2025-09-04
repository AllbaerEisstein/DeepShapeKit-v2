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
from .LBS import LBS
#from .LBS_origin import LBS


class fish_model():
    """
    Parametric fish model.
    !! Properties will only be on the device after calling to_device() !!
    Input:
        mesh: path to json file containing fish model
        device: 'cpu' or 'cuda'
    Output:
        faces: (F, 3) faces of the fish mesh
        J: (1, J, 3) joint locations in the rest pose
        V: (1, V, 3) vertices in the rest pose
        vert2kpt: (K, V) keypoint regression matrix
        weights: (V, J) skinning weights
    On call:
    """
    def __init__(self, mesh, device: str):
        self.device = torch.device(device)
        self.device_active = False

        with open(mesh, 'r') as infile:
            dd = json.load(infile)

        self.dd = dd

        self.n_bones = dd["n_bones"]

        # triangulate if input mesh has quad faces
        fish_faces = dd['F']
        triangle_faces = []
        for face in fish_faces:
            if len(face) == 4:
                triangle_faces.append([face[0], face[1], face[2]])
                triangle_faces.append([face[2], face[3], face[0]])
            elif len(face) == 3:
                triangle_faces.append(face)
            else:
                raise Exception('face data incorrect: got vertices number not in [3,4]')
        self.faces = torch.tensor(triangle_faces)

        # self.faces = torch.tensor(dd['F'])

        self.kintree_table = torch.tensor(dd['kintree_table'])[:,:]
        self.parents = self.kintree_table[0][:]

        self.weights = torch.tensor(dd['weights'])
        self.vert2kpt = torch.tensor(dd['vert2kpt'])

        # TODO: there used to be an unsqueeze(0) added and output V used to be (1, V, 3) - same for J
        self.J = torch.tensor(dd['J']).unsqueeze(0)

        self.V = torch.tensor(dd['V']).unsqueeze(0)
        self.V = self.V - self.J[0,0]
        self.J = self.J - self.J[0,0]

        self.V = self.V * 0.01
        self.J = self.J * 0.01

        self.LBS = LBS(self.J, self.parents, self.weights)
    
    def to_device(self, device: str):
        self.device = torch.device(device)
        self.kintree_table = self.kintree_table.to(device)
        self.parents = self.parents.to(device)
        self.faces = self.faces.to(self.device)
        self.weights = self.weights.to(self.device)
        self.vert2kpt = self.vert2kpt.to(self.device)
        self.J = self.J.to(self.device)
        self.V = self.V.to(self.device)
        if not all(self.faces.device == attr.device for attr in [self.kintree_table, self.parents, self.weights, self.vert2kpt, self.J, self.V]):
            raise ValueError("failed to move all fish object attributes to the same device")
        self.LBS = LBS(self.J, self.parents, self.weights)
        self.device = self.faces.device
        self.device_active = True

    def __call__(self, global_pose, body_pose, bone_length, scale=1, pose2rot=True):
        """
        Input:
            global_pose: (B, 3) axis-angle representation of global rotation
            body_pose: (B, P*3) axis-angle representation of body pose (exclude root joint orient)
            bone_length: (B, B) bone length
            scale: (B, 1) scale factor
            pose2rot: if True, convert axis-angle to rotation matrix inside LBS
        """
        assert all(self.faces.device == attr.device for attr in [global_pose, body_pose, bone_length]), "All inputs must be on the same device as specified"
        
        batch_size = global_pose.shape[0]
        V = self.V.repeat([batch_size, 1, 1]) * scale

        # concatenate bone and pose
        bone = torch.cat([torch.ones([batch_size, 1], device=self.device), bone_length], dim=1)
        pose = torch.cat([global_pose, body_pose], dim=1)

        # LBS
        verts = self.LBS(V, pose, bone, scale, to_rotmats=pose2rot)

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = []
        for i in range(verts.shape[0]):
            kpt = torch.matmul(self.vert2kpt, verts[i])
            keypoints.append(kpt)
        keypoints = torch.stack(keypoints)

        # Final output after articulation
        output = {'vertices': verts.cpu(),
                  'keypoints': keypoints.cpu()}

        return output
