"""
linear blend skinning from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""

import torch
from torch.nn import functional as F

from src.geometry import batch_rodrigues


class LBS():
    '''
    Implementation of linear blend skinning, with additional bone and scale
    Args:
        V (BN, V, 3): vertices to pose and shape
        pose (BN, J, 3, 3) or (BN, J, 3): pose in rot or axis-angle
        bone (BN, K): allow for direct change of relative joint distances
        scale (1): scale the whole kinematic tree
    '''

    def __init__(self, J, parents, weights):
        """
        Args:
            parents (BS, jn): List of parent indices for jn joints for BS batches. Parent index of the first joint is -1 (no parent)
        """
        self.n_joints = J.shape[1] - 1
        # rotation around joint
        # self.h_joints = F.pad(J[:,1:].unsqueeze(-1), [0, 0, 0, 1], value=0)
        # #self.kin_tree = torch.cat([J[:, [0], :], J[:, 1:] - J[:, parents[1:]]], dim=1).unsqueeze(-1)
        # self.kin_tree = (J[:, 1:] - J[:, parents[1:]]).unsqueeze(-1)
        # print(f"Joints shape: {J.size()}")
        # rotation around parent
        self.h_joints = F.pad((J[:, parents[1:]]).unsqueeze(-1), [0, 0, 0, 1], value=0)
        # kin_tree: positions of the body joints (first joint excluded!) relative to their parents
        self.kin_tree = (J[:, 1:] - J[:, parents[1:]]).unsqueeze(-1)
        self.joint_parent = F.pad((J[:, parents[1:]]).unsqueeze(-1), [0,0,0,1], value=1)

        self.parents = parents
        #self.parents[0] = -1
        self.weights = weights[None].float()

    def __call__(self, V, pose, bone, scale, to_rotmats=True):
        """
        Args:
            V (BS, vn, 3): 3d coordinates (world) of vn vertices in BS batches
            pose (BS, bn, 3): axis-angle poses of all bn bones of the mesh in BS batches
            scale (1): scalar denoting the size factor
        """
        batch_size = len(V)
        device = pose.device
        V = F.pad(V.unsqueeze(-1), [0, 0, 0, 1], value=1)
        # TODO: used to cut the first bone: bone[:, 1:, None, None]
        # print(f"LBS: kintree shape: {self.kin_tree.size()}")
        # print(f"LBS: bone length shape: {bone.size()}")
        # print(f"LBS: pose shape: {pose.size()}")
        kin_tree = (scale * self.kin_tree) * bone[:, :, None, None]

        # disable rotation around x-axis for fish parts
        for i in range(self.n_joints - 1):
            pose[0,3 * (i + 1)] = 0.

        if to_rotmats:
            pose = batch_rodrigues(pose.view(-1, 3))
        pose = pose.view([batch_size, -1, 3, 3])
        T = torch.zeros([batch_size, self.n_joints, 4, 4]).float().to(device)
        T[:, :, -1, -1] = 1
        # TODO: used to cut the first pose: pose[:,1:,:,:]
        T[:, :, :3, :] = torch.cat([pose[:,:,:,:], kin_tree], dim=-1)
        T_rel = [T[:, 0]]
        for i in range(1, self.n_joints):
            T_rel.append(T_rel[self.parents[i]] @ T[:, i])
        T_rel = torch.stack(T_rel, dim=1)
        T_rel[:, :, :, [-1]] -= T_rel.clone() @ (self.h_joints * scale)

        # rotation around parent
        # parent_pos = T_rel.clone() @ (self.joint_parent * scale) - T_rel[:, [0,0,1,2,3], :, :].clone() @ (self.joint_parent * scale)
        # parent_pos[:,1:,:,:] += parent_pos[:,[0],:,:]
        # parent_pos[:, 2:, :, :] += parent_pos[:, [1], :, :]
        # parent_pos[:, 3:, :, :] += parent_pos[:, [2], :, :]
        # parent_pos[:, 4:, :, :] += parent_pos[:, [3], :, :]
        # T_rel[:, :, :, [-1]] += parent_pos

        T_ = self.weights @ T_rel.view(batch_size, self.n_joints, -1)
        T_ = T_.view(batch_size, -1, 4, 4)

        V = T_ @ V

        R = torch.zeros([batch_size, 1, 4, 4]).float().to(device)
        R[:, :, :3, :3] = pose[:,[0],:,:]
        R[:, :, -1, -1] = 1
        V = R @ V

        return V[:, :, :3, 0] / V[:, :, [3], 0]
