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

    def __init__(self, J, parent_indices, weights, virtual_bone_mask):
        """
        Args:
            J (BN, JN, 3): joint positions in local space for BN batches. JN should be the total joint number, so n_bones+1 (there is one more joint (head) than bones)
            parents (JN): list of indices of the parent of every joint (e.g. [-1,0,1,2,3,4]) for 6 joints
            weights (n_vertices, n_bones):
            virtual_bone_mask (n_bones): binary mask for which bones are virtual bones (virtual bone at index i: mask[i]=1). Virtual bone rotation is set to identity in the LBS-call since virtual bones represent a translation of parent joint to child joint only.

            !! 
            Attention: Everything passed to LBS is treated to be in the same reference as the Joints.
            Usually, this is the same space the mesh is defined in, a.k.a., the template space.
            Even if something is "relative to head", it still doesn't reference the head bone coordinate system.
            !!
        """
        self.J = J
        # the head joint is just the origin of the mesh. it is excluded
        self.n_body_joints = J.shape[1] - 1

        # CLAUDE FIX: `joints_homog` holds the rest position of *the joint each transform belongs to*.
        # Entry k of the transform chain describes joint k+1 (the head joint is excluded), so the
        # rest-pose reference for entry k must be J[k+1] -- i.e. `J[:, 1:]`. This is what makes the
        # skinning transform reduce to the identity at zero pose, so that calling the model with a
        # zero pose, unit bone lengths and unit scale returns the template mesh unchanged.
        # (See `assert_rest_pose_is_identity()` below, which asserts exactly that.)
        self.joints_homog = F.pad((J[:, 1:]).unsqueeze(-1), [0, 0, 0, 1], value=0)
        # body_joint_locs_rel_to_parents: positions of the body joints (first joint excluded!) relative to their parents
        # ---> len(locs_rel_to_parents) = n_joints-1 = n_bones
        self.body_joint_locs_rel_to_parents = (J[:, 1:] - J[:, parent_indices[1:]]).unsqueeze(-1)

        self.parent_indices = parent_indices

        # CLAUDE FIX: the kinematic tree is walked in index order, so every joint must appear after
        # its parent. Validate this once here instead of silently producing a garbled skeleton.
        for joint_idx in range(1, J.shape[1]):
            parent = int(parent_indices[joint_idx])
            if parent >= joint_idx or parent < 0:
                raise ValueError(
                    f"kintree_table is not topologically ordered: joint {joint_idx} has parent "
                    f"{parent}. Every joint must be listed after its parent, and only joint 0 "
                    f"(the head) may have parent -1."
                )

        self.weights = weights[None].float()

        # CLAUDE FIX: the homogeneous divide at the end of the skinning pass normalizes each vertex
        # by the sum of its skinning weights, so unnormalized weights are fine -- but a vertex with
        # a total weight of zero would divide by zero and poison the whole optimization with NaNs.
        # Reject such templates at load time with an actionable message.
        weight_sums = self.weights[0].sum(dim=1)
        if not torch.isfinite(weight_sums).all() or (weight_sums.abs() < 1e-8).any():
            bad = int((weight_sums.abs() < 1e-8).sum())
            raise ValueError(
                f"{bad} vertex/vertices in the template have a total skinning weight of ~0. "
                f"Every vertex must be influenced by at least one bone; re-export the template "
                f"with complete skinning weights."
            )

        self.virtual_bone_mask = virtual_bone_mask

    def assert_rest_pose_is_identity(self, V: torch.Tensor, tol: float = 1e-4) -> float:
        """
        CLAUDE FIX: regression guard for the skinning transform.

        Skinning with a zero pose, unit bone lengths and unit scale must return the template mesh
        unchanged. If this does not hold, every keypoint and mask residual is being fitted against a
        distorted forward model, which is silent and hard to spot in rendered output.

        Returns the maximum per-vertex deviation and raises if it exceeds `tol`.
        """
        n_bones = self.weights.shape[-1]
        with torch.no_grad():
            zero_pose = torch.zeros(V.shape[0], n_bones * 3, device=V.device, dtype=V.dtype)
            unit_lengths = torch.ones(V.shape[0], n_bones, device=V.device, dtype=V.dtype)
            unit_scale = torch.ones(1, device=V.device, dtype=V.dtype)
            verts, _ = self(V, zero_pose, unit_lengths, unit_scale)
            max_dev = float((verts - V).abs().max())
        if max_dev > tol:
            raise ValueError(
                f"LBS does not reproduce the rest pose: max vertex deviation {max_dev:.6f} > {tol}. "
                f"The skinning transform and the template are inconsistent."
            )
        return max_dev

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

        if not torch.isfinite(global_ori_plus_body_pose).all():
            raise ValueError("LBS received non-finite pose values before Rodrigues conversion.")
        if not torch.isfinite(all_bone_length).all():
            raise ValueError("LBS received non-finite bone lengths.")
        if not torch.isfinite(scale).all():
            raise ValueError("LBS received non-finite scale.")

        V_homog = F.pad(V.unsqueeze(-1), [0, 0, 0, 1], value=1)
        
        # scale the joint positions by the bone lengths -> init kin-tree excluded head joint, this step should exclude it, too
        # however, the head bone length is important since the first *body joint* has a location relative to its 
        # parent (head joint) that needs to be scaled.
        body_joint_locs_rel_to_parents = (scale * self.body_joint_locs_rel_to_parents) * all_bone_length[:, :, None, None]

        global_ori_plus_body_pose = batch_rodrigues(global_ori_plus_body_pose.view(-1, 3), to_rotmats=True, to_quats=False) # view: pose from list format ([[a,b,c,a1,b1,c1,...]]) to triplet format ([[[a,b,c],[a1,b1,c1],...]])
        global_ori_plus_body_pose = global_ori_plus_body_pose.view([batch_size, -1, 3, 3])
        # hard-code virtual bone rotation to identity (no rotation) since virtual bones do not have a pose and should not contribute to the deformation of the mesh
        global_ori_plus_body_pose[:, self.virtual_bone_mask.bool(), :, :] = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)

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
        # CLAUDE FIX: two index spaces meet here and must be converted between explicitly.
        # `parent_indices` is indexed by *joint* index (0 = head), while this list is indexed by
        # *body-joint* index, where entry k describes joint k+1. So the parent of the joint at list
        # position i is joint `parent_indices[i + 1]`, which lives at list position
        # `parent_indices[i + 1] - 1`. A parent of joint 0 (the head) has no list entry and means the
        # joint hangs directly off the root, so its chained transform is its own relative transform.
        # For a straight chain both index spaces happen to agree; for any branching rig (fins,
        # limbs) they do not, so subtrees would otherwise be attached to the wrong parent.
        for i in range(1, self.n_body_joints):
            parent_joint_idx = int(self.parent_indices[i + 1])
            parent_list_idx = parent_joint_idx - 1
            if parent_list_idx < 0:
                # joint (i+1) is a direct child of the head joint
                T_for_joints_rel_to_head.append(T_for_joints_rel_to_parent[:, i])
            else:
                T_for_joints_rel_to_head.append(
                    T_for_joints_rel_to_head[parent_list_idx] @ T_for_joints_rel_to_parent[:, i]
                )
        T_for_joints_rel_to_head = torch.stack(T_for_joints_rel_to_head, dim=1)
        T_for_joints_rel_to_head[:, :, :, [-1]] -= T_for_joints_rel_to_head.clone() @ (self.joints_homog * scale)
        #   compound_rotmat(0,0) compound_rotmat(0,1) compound_rotmat(0,2) transformed_joint[0]
        #   compound_rotmat(1,0) compound_rotmat(1,1) compound_rotmat(1,2) transformed_joint[1]
        #   compound_rotmat(2,0) compound_rotmat(2,1) compound_rotmat(2,2) transformed_joint[2]
        #           0                    0                    0                   1
        # ---> the translation of each joint is now offset by the position of this joint.
        #      future vertex translation depends on associated joint position.

        # CLAUDE FIX: the bone-angle (swing-twist) priors are per-joint articulation limits, so they
        # must be evaluated on each bone's *own* rotation relative to its parent -- not on the
        # accumulated rotation of the whole chain up to that bone, which is what
        # `T_for_joints_rel_to_head[..., :3, :3]` contains. Returning the local rotations means a
        # long spine of individually legal bends is no longer reported as a large violation, and
        # equal-and-opposite bends no longer cancel out and escape the limits.
        # Index 0 of this tensor is the global orientation (the root bone's own rotation), which is
        # applied separately below; indices 1.. are the body bones, in bone order.
        local_bone_rotations = global_ori_plus_body_pose  # (BS, n_bones, 3, 3)

        T_for_vertices = self.weights @ T_for_joints_rel_to_head.view(batch_size, self.n_body_joints, -1)
        T_for_vertices = T_for_vertices.view(batch_size, -1, 4, 4)

        V_homog = T_for_vertices @ V_homog

        # apply the global orientation (orientation of first bone):
        R = torch.zeros([batch_size, 1, 4, 4]).float().to(device)
        R[:, :, :3, :3] = global_ori_plus_body_pose[:,[0],:,:]
        R[:, :, -1, -1] = 1
        V_homog = R @ V_homog

        # CLAUDE FIX: the homogeneous coordinate here is the sum of each vertex's skinning weights,
        # so this divide is what normalizes unnormalized weights. Clamp its magnitude so a
        # near-degenerate vertex degrades gracefully instead of producing NaNs that propagate into
        # every loss term. (Exactly-zero weight sums are rejected in __init__.)
        w_homog = V_homog[:, :, [3], 0]
        w_sign = torch.where(w_homog < 0, -torch.ones_like(w_homog), torch.ones_like(w_homog))
        w_homog = w_sign * w_homog.abs().clamp_min(1e-8)

        # return vertices in local space (relative to head joint) and each bone's own (local)
        # rotation in template space (for bone angle prior checking)
        return V_homog[:, :, :3, 0] / w_homog, local_bone_rotations