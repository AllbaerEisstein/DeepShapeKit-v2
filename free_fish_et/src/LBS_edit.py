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
    '''

    def __init__(self, J, parent_indices, weights):
        """
        Args:
            J (BN, JN, 3): joint positions in local space for BN batches. JN should be the total joint number, so n_bones+1 (there is one more joint (head) than bones)
            parents (JN): list of indices of the parent of every joint (e.g. [-1,0,1,2,3,4]) for 6 joints
            weights (n_vertices, n_bones): 
        """
        self.J = J
        # the head joint is just the origin of the mesh. it is excluded
        self.n_body_joints = J.shape[1] - 1

        self.joints_homog = F.pad((J[:, parent_indices[1:]]).unsqueeze(-1), [0, 0, 0, 1], value=0)
        # body_joint_locs_rel_to_parents: positions of the body joints (first joint excluded!) relative to their parents
        # ---> len(locs_rel_to_parents) = n_joints-1 = n_bones
        self.body_joint_locs_rel_to_parents = (J[:, 1:] - J[:, parent_indices[1:]]).unsqueeze(-1)

        self.parent_indices = parent_indices
        self.weights = weights[None].float()

    def __call__(self, V, global_ori_plus_body_pose, all_bone_length, scale, to_rotmats=True):
        """
        Args:
            V (BS, vn, 3): 3d coordinates (local) of vn vertices in BS batches
            global_ori_plus_body_pose (BS, bn*3): exponential map (axis-angle where length of the axis vector denotes angle) poses of all bones (first bone included) of the mesh in BS batches
            all_bone_length (BS, bn): length of bn bones
            scale (1): scalar denoting the size factor
        """
        batch_size = len(V)
        device = global_ori_plus_body_pose.device
        V_homog = F.pad(V.unsqueeze(-1), [0, 0, 0, 1], value=1)
        
        # scale the joint positions by the bone lengths -> init kin-tree excluded head joint, this step should exclude it, too
        # however, the head bone length is important since the first *body joint* has a location relative to its 
        # parent (head joint) that needs to be scaled.
        body_joint_locs_rel_to_parents = (scale * self.body_joint_locs_rel_to_parents) * all_bone_length[:, :, None, None]

        if to_rotmats:
            global_ori_plus_body_pose = batch_rodrigues(global_ori_plus_body_pose.view(-1, 3)) # view: pose from list format ([[a,b,c,a1,b1,c1,...]]) to triplet format ([[[a,b,c],[a1,b1,c1],...]])
        global_ori_plus_body_pose = global_ori_plus_body_pose.view([batch_size, -1, 3, 3])

        T_for_joints_rel_to_parent = torch.zeros([batch_size, self.n_body_joints, 4, 4]).float().to(device) # 4x4 all-0 matrices for every joint (excluding head joint)
        T_for_joints_rel_to_parent[:, :, -1, -1] = 1 # last-row last-column entry (bottom right) is 1 (homogeneous transform)

        # first body joint just gets translated via bone_length-scaling; no rotation
        T_for_joints_rel_to_parent[:, 0, :3, :]  = torch.cat([torch.eye(3, device=device).unsqueeze(0), body_joint_locs_rel_to_parents[:,0,:]], dim=-1)
        # now, T looks like this for the first body joint:
        #       1           0           0       loc_rel_to_parent[0]
        #       0           1           0       loc_rel_to_parent[1]
        #       0           0           1       loc_rel_to_parent[2]
        #       0           0           0           1
        T_for_joints_rel_to_parent[:, 1:, :3, :] = torch.cat([global_ori_plus_body_pose[:,1:,:,:], body_joint_locs_rel_to_parents[:,1:,:]], dim=-1)
        # now, T looks like this for every joint (excluding first two joints, i.e. excluding head joint and first body joint):
        #
        #   rotmat(0,0) rotmat(0,1) rotmat(0,2) loc_rel_to_parent[0]
        #   rotmat(1,0) rotmat(1,1) rotmat(1,2) loc_rel_to_parent[1]
        #   rotmat(2,0) rotmat(2,1) rotmat(2,2) loc_rel_to_parent[2]
        #       0           0           0           1
        #
        # ---> so it is a transformartion matrix that performs the relative rotation specified by pose and 
        #      translation to the position of the joint relative to its parent

        T_for_joints_rel_to_head = [T_for_joints_rel_to_parent[:, 0]] 
        # this will be a recursively generated transformation for each joint (excluding head) built by 
        # successively applying all the transformations of this joints parents
        # ---> so T_rel_chained will have transformations relative to the head joint for each joint
        #      (because every transformation will build on the transformation of the first body joint and 
        #      that transf is relative to the head joint)
        for i in range(1, self.n_body_joints):
            T_for_joints_rel_to_head.append(T_for_joints_rel_to_head[self.parent_indices[i]] @ T_for_joints_rel_to_parent[:, i]) # self.parent_indices[i] retrieves current joint's parent, means we get the transformation matrix of this joint's parent
        T_for_joints_rel_to_head = torch.stack(T_for_joints_rel_to_head, dim=1)
        T_for_joints_rel_to_head[:, :, :, [-1]] -= T_for_joints_rel_to_head.clone() @ (self.joints_homog * scale)
        #   compound_rotmat(0,0) compound_rotmat(0,1) compound_rotmat(0,2) transformed_joint[0]
        #   compound_rotmat(1,0) compound_rotmat(1,1) compound_rotmat(1,2) transformed_joint[1]
        #   compound_rotmat(2,0) compound_rotmat(2,1) compound_rotmat(2,2) transformed_joint[2]
        #           0                    0                    0                   1
        # ---> the translation of each joint is now offset by the position of this joint.
        #      future vertex translation depends on associated joint position.

        T_for_vertices = self.weights @ T_for_joints_rel_to_head.view(batch_size, self.n_body_joints, -1)
        T_for_vertices = T_for_vertices.view(batch_size, -1, 4, 4)

        V_homog = T_for_vertices @ V_homog

        # apply the global orientation (orientation of first bone):
        R = torch.zeros([batch_size, 1, 4, 4]).float().to(device)
        R[:, :, :3, :3] = global_ori_plus_body_pose[:,[0],:,:]
        R[:, :, -1, -1] = 1
        V_homog = R @ V_homog

        return V_homog[:, :, :3, 0] / V_homog[:, :, [3], 0]
