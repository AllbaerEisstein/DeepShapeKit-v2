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

        # CLAUDE FIX (BUGREPORT A10, revised): entry k of the transform chain is BONE k, and a bone
        # rotates about its own HEAD joint, which is J[parent_indices[k+1]].
        #
        # A previous revision restored the rest pose by moving this to `J[:, 1:]` (the bone's TAIL)
        # while leaving the offset below as the bone's own rest vector. That combination is
        # self-consistent -- it passes assert_rest_pose_is_identity -- but it is SMPL *joint*
        # semantics: because the offset sits in T_k next to R_k, it is rotated by the chain WITHOUT
        # R_k, so pose[k] never moves bone k's own tail and every child of one bone is forced to
        # share a single orientation. A Blender armature and this model then cannot be made to
        # agree for any choice of pose (see assert_bone_rotates_its_own_segment below).
        #
        # The head pivot is paired with the parent-bone offset in __call__; together they keep the
        # rest pose exact AND make pose[k] the rotation of bone k about its own head, which is what
        # the Blender vertex groups, the per-bone swing-twist priors and pose_time_series/2 all
        # assume.
        self.joints_homog = F.pad((J[:, parent_indices[1:]]).unsqueeze(-1), [0, 0, 0, 1], value=0)
        # body_joint_locs_rel_to_parents: positions of the body joints (first joint excluded!) relative to their parents
        # ---> len(locs_rel_to_parents) = n_joints-1 = n_bones
        # entry k is bone k's OWN rest vector (head -> tail); __call__ gathers the PARENT's entry.
        self.body_joint_locs_rel_to_parents = (J[:, 1:] - J[:, parent_indices[1:]]).unsqueeze(-1)

        self.parent_indices = parent_indices
        # parent BONE of bone k, -1 for the root bone. Bone k's head is its parent bone's tail, so
        # joint parent_indices[k+1] is bone (parent_indices[k+1] - 1)'s tail.
        self.parent_bone = [int(parent_indices[k + 1]) - 1 for k in range(self.n_body_joints)]

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

    def assert_bone_rotates_its_own_segment(self, tol: float = 1e-3) -> float:
        """
        CLAUDE FIX (BUGREPORT A10): companion guard to assert_rest_pose_is_identity.

        Rotating bone k must displace bone k's own tail. Both the head-pivot and the SMPL joint
        variant of the skinning chain reproduce the rest pose, so assert_rest_pose_is_identity
        cannot tell them apart; this one can. Under the joint variant the skeleton does not move at
        all when a bone is rotated, and no exporter can make a Blender armature and this model
        agree.

        Returns the tail displacement produced by a 0.9 rad bend and raises if it is ~0.
        """
        n_bones = self.n_body_joints
        if n_bones < 2:
            return float("inf")
        # pick the last non-virtual body bone (virtual bones are forced to identity in __call__)
        probe = None
        for k in range(n_bones - 1, 0, -1):
            if not bool(self.virtual_bone_mask[k]):
                probe = k
                break
        if probe is None:
            return float("inf")

        device = self.J.device
        with torch.no_grad():
            lengths = torch.ones(1, n_bones, device=device)
            scale = torch.ones(1, device=device)
            tails = []
            for angle in (0.0, 0.9):
                pose = torch.zeros(1, n_bones * 3, device=device)
                pose[0, 3 * probe + 2] = angle
                G = self._transform_chain(self._rotmats(pose, 1), lengths, scale)
                head = G[0, probe, :3, 3]
                rot = G[0, probe, :3, :3]
                own_vec = self.body_joint_locs_rel_to_parents.to(device)[0, probe, :, 0]
                tails.append(head + rot @ (own_vec * scale))
            moved = float((tails[1] - tails[0]).norm())
        if moved < tol:
            raise ValueError(
                f"LBS: rotating bone {probe} displaced its own tail by {moved:.3e} (< {tol}). "
                f"pose[k] is seated at bone k's tail instead of its head, so a Blender armature "
                f"and this model cannot be made to agree. See BUGREPORT A10."
            )
        return moved

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

    def _rotmats(self, global_ori_plus_body_pose, batch_size):
        """Exponential-map pose -> (BS, n_bones, 3, 3), with virtual bones pinned to identity."""
        device = global_ori_plus_body_pose.device
        R = batch_rodrigues(global_ori_plus_body_pose.view(-1, 3), to_rotmats=True, to_quats=False)
        R = R.view([batch_size, -1, 3, 3])
        # hard-code virtual bone rotation to identity (no rotation) since virtual bones do not have
        # a pose and should not contribute to the deformation of the mesh
        R[:, self.virtual_bone_mask.bool(), :, :] = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
        return R

    def _transform_chain(self, rotmats, all_bone_length, scale):
        """
        Chained bone transforms G_k (BS, n_bones, 4, 4), whose translation column is bone k's posed
        HEAD position (the rest-pose offset is removed by the caller).

        Args:
            rotmats (BS, n_bones, 3, 3): per-bone rotations, as produced by `_rotmats`
        """
        R = rotmats
        device = R.device
        batch_size = R.shape[0]
        n_bones = self.n_body_joints

        # CLAUDE FIX (BUGREPORT A10, revised): the offset that separates bone k from its parent is
        # the PARENT bone's rest vector, scaled by the PARENT's length factor -- not bone k's own
        # vector. Inside the chain G_k = G_parent @ T_k the offset is rotated by G_parent only, so
        # putting bone k's own vector there means R_k never moves bone k's tail (and all siblings
        # inherit one shared orientation). The root bone has no parent offset.
        parent_bone = torch.tensor(self.parent_bone, device=device, dtype=torch.long)
        gather = parent_bone.clamp_min(0)
        valid = (parent_bone >= 0).to(torch.float32).view(1, n_bones, 1, 1)
        locs = self.body_joint_locs_rel_to_parents.to(device)[:, gather]          # (1, nb, 3, 1)
        offsets = (scale * locs) * all_bone_length[:, gather][:, :, None, None] * valid

        T_rel = torch.zeros([batch_size, n_bones, 4, 4], device=device).float()
        T_rel[:, :, -1, -1] = 1
        T_rel[:, :, :3, :] = torch.cat([R, offsets.expand(R.shape[0], -1, -1, -1)], dim=-1)
        # bone 0's rotation is the global orientation; it is applied to the whole mesh at the end
        T_rel[:, 0, :3, :3] = torch.eye(3, device=device).unsqueeze(0)

        # CLAUDE FIX: two index spaces meet here and must be converted between explicitly.
        # `parent_indices` is indexed by *joint* index (0 = head), while this list is indexed by
        # *bone* index, where entry k describes bone k / joint k+1. `self.parent_bone[k]` performs
        # that conversion once, in __init__. For a straight chain both index spaces happen to
        # agree; for any branching rig (fins, limbs) they do not, so subtrees would otherwise be
        # attached to the wrong parent.
        chain = [T_rel[:, 0]]
        for i in range(1, n_bones):
            parent = self.parent_bone[i]
            chain.append(T_rel[:, i] if parent < 0 else chain[parent] @ T_rel[:, i])
        return torch.stack(chain, dim=1)

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

        # rotation matrices per bone (virtual bones pinned to identity); index 0 is the global
        # orientation, which is applied to the whole mesh further down rather than inside the chain
        global_ori_plus_body_pose = self._rotmats(global_ori_plus_body_pose, batch_size)

        # chained transforms; the translation column of entry k is bone k's posed HEAD
        T_for_joints_rel_to_head = self._transform_chain(
            global_ori_plus_body_pose, all_bone_length, scale)
        # remove the rest-pose position of each bone's pivot (its head joint), so that the
        # transform reduces to the identity when the pose is zero
        T_for_joints_rel_to_head = T_for_joints_rel_to_head.clone()
        T_for_joints_rel_to_head[:, :, :, [-1]] -= (
            T_for_joints_rel_to_head.clone() @ (self.joints_homog.to(device) * scale)
        )

        # CLAUDE FIX: the bone-angle (swing-twist) priors are per-joint articulation limits, so they
        # must be evaluated on each bone's *own* rotation relative to its parent -- not on the
        # accumulated rotation of the whole chain up to that bone. Returning the local rotations
        # means a long spine of individually legal bends is no longer reported as a large
        # violation, and equal-and-opposite bends no longer cancel out and escape the limits.
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