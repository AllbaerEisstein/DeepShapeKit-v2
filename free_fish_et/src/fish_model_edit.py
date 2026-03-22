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
from src.geometry import batch_rodrigues
import torch
from .LBS_edit import LBS
from pytorch3d.transforms import matrix_to_quaternion

# from .LBS_origin import LBS


class fish_model:
    """
    Parametric fish model.
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
        """
        self.device = torch.device("cpu")
        self.device_active = False

        with open(mesh_json_path, "r") as infile:
            template = json.load(infile)

        self.template = template

        self.n_bones = template["n_bones"]
        # the first bone is treated differently from all the other bones (first bone: global orientation fitting; body bones: pose fitting)
        # So n_body_bones excludes the first bone
        self.n_body_bones = template["n_bones"] - 1

        # we always have one more joint than bone (the head)
        self.n_joints = len(template["J"])

        # triangulate if input mesh has quad faces
        fish_faces = template["F"]
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

        self.kintree_table = torch.tensor(template["kintree_table"])[:, :]
        self.parent_indices = self.kintree_table[0][:]

        # a bone group is a set of indices for bones and keypoints that are going to be reconstructed together in one optimizer stage
        # keypoints and bones belonging to group 0 are at keypoint_groups[0] and bone_groups[0], respectively
        self.bone_groups = [bg["bone_indices"] for bg in template["bone_groups"]]
        self.keypoint_groups = [bg["keypoint_indices"] for bg in template["bone_groups"]]

        self.weights = torch.tensor(template["weights"])
        self.vert2kpt = torch.tensor(template["vert2kpt"])

        self.J = torch.tensor(template["J"], dtype=torch.float32).unsqueeze(0) # (1,J,3)
        self.virtual_bone_mask = torch.tensor(template["virtual_bone_mask"]) # (n_bones) binary mask for which bones are virtual bones (virtual bone at index i: mask[i]=1)
        # set the mesh local space to head space (head joint at origin; head-bone local axes aligned with world axes)
        # use rest_rot_world from the mesh json.

        def get_rot_matrix_for_y_axis_rotation(new_y_axis):
            """
            This function calculates the rotation matrix that rotates the positive y-axis to the new y-axis (bone twist axis).
            """
            # take care of the cases in that y-axis is parallel to the head-to-first-body-joint direction
            if torch.isclose(new_y_axis/torch.norm(new_y_axis), torch.tensor([0, -1, 0], dtype=torch.float32)).all():
                rot_matrix_passive = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)
            elif not torch.isclose(new_y_axis/torch.norm(new_y_axis), torch.tensor([0, 1, 0], dtype=torch.float32)).all():
                rot_angle = torch.acos(torch.dot(new_y_axis, torch.tensor([0, 1, 0], dtype=torch.float32)) / torch.norm(new_y_axis))
                rot_axis = torch.linalg.cross(new_y_axis, torch.tensor([0, 1, 0], dtype=torch.float32))
                rot_axis = rot_axis / torch.norm(rot_axis)
                axis_angle = rot_axis * rot_angle
                # convert axis-angle to rotation matrix using Rodrigues' formula
                rot_matrix_passive = batch_rodrigues(axis_angle.unsqueeze(0), to_quats=False, to_rotmats=True)[0]
            else:
                rot_matrix_passive = torch.eye(3, dtype=torch.float32)
            # This is the rotation the y-axis has to undergo (passive rotation).
            # To express a point in the coordinate system with the new y-axis, 
            # we need to apply the inverse of this rotation (active rotation) to the point.
            return rot_matrix_passive
        
        def get_root_rest_rotation_world():
            bone_tree = template.get("bone_names_tree", {})

            root_bone_name = template["bone_order"][0]
            if root_bone_name not in bone_tree:
                raise ValueError(f"Root bone {root_bone_name} does not have a corresponding entry in the bone tree of the json file. Please make sure that the root bone has an entry in the bone tree and that the bone tree is correctly formatted.")

            root_data = bone_tree[root_bone_name]
            rest_rot_rows = root_data.get("rest_rot_world", None)

            R = torch.tensor(rest_rot_rows, dtype=torch.float32)
            # Re-orthonormalize to guard against tiny numerical drift in serialized matrices.
            U, _, Vh = torch.linalg.svd(R)
            R = U @ Vh
            if torch.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vh
            return R

        # translate to head-local coords (head joint -> origin), then rotate model/world frame -> head frame
        translation_head_to_origin = self.J[0, 0].unsqueeze(0)
        root_rest_rot_world = get_root_rest_rotation_world()

        # rest_rot_world maps head-local axes to world/model axes (local->world).
        # We need the inverse to express points in head-local coordinates.
        R_model_space_to_head_space = root_rest_rot_world.T

        self.J = self.J - translation_head_to_origin
        self.J = torch.matmul(self.J, R_model_space_to_head_space)

        self.V = torch.tensor(template["V"]).unsqueeze(0) # (1,V,3)
        # apply the same rotation and translation to the vertices as we did to the joints in order to keep them in the same local space
        self.V = torch.matmul(self.V - translation_head_to_origin, R_model_space_to_head_space)

        body_bone_twist_axes_head_space = (self.J[:, 2:] - self.J[:, self.parent_indices[2:]])
        # exclude the head bone since we treat head bone transformation as global transformation and do not calculate swing-twist for the head bone
        rot_mats_to_body_bone_rest_local_spaces = [] 
        for i in range(self.n_body_bones): 
            twist_axis = body_bone_twist_axes_head_space[:, i]
            rot_mats_to_body_bone_rest_local_spaces.append(get_rot_matrix_for_y_axis_rotation(twist_axis.squeeze(0)).T)
        # when calculating the swing-twist of each bone, we need to express each articulated bone tail 
        # in the local space of the rest bone head because the twist axis is required to be one of x,y, or z.
        self.rot_mats_to_body_bone_rest_local_spaces = torch.stack(rot_mats_to_body_bone_rest_local_spaces, dim=0) # (n_body_bones, 3, 3)

        # # scaling, unit conversion
        # self.V = self.V #* 0.01
        # self.J = self.J #* 0.01

        self.LBS = LBS(self.J, self.parent_indices, self.weights, self.virtual_bone_mask)

        # global_ori and body_pose angle limits (import priors from fish template json)
        # angle limits are specified for each component in swing-twist representation (swing_x, twist_y, swing_z) (in radian) for each bone
        # where swing_x and swing_z are x- and y- radii of an ellipse that bounds the swing rotation, and twist_y is the maximum allowed twist rotation around the y-axis.
        bone_angle_priors = []
        for bname in template["bone_order"]:
            if bname not in template["bone_priors"]:
                raise ValueError(f"Bone {bname} does not have a corresponding prior in the json file. Please make sure that all bones have priors specified.")
            prior = template["bone_priors"][bname]
            bone_angle_priors.extend([prior["swing_x"], prior["twist_y"], prior["swing_z"]])
        self.bone_angle_priors = torch.tensor(bone_angle_priors).view(1, -1, 3) # (1, n_bones, 3) (swing_x, twist_y, swing_z)

        # Body bone length limit
        self.bone_length_min = [1.0] * (self.n_body_bones)
        self.bone_length_max = [1.5] * (self.n_body_bones)

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
        self.bone_angle_priors = self.bone_angle_priors.to(self.device)
        self.bone_length_min = self.bone_length_min.to(self.device)
        self.bone_length_max = self.bone_length_max.to(self.device)
        self.virtual_bone_mask = self.virtual_bone_mask.to(self.device)
        self.LBS = LBS(self.J, self.parent_indices, self.weights, self.virtual_bone_mask)
        self.device = self.faces.device
        self.rot_mats_to_body_bone_rest_local_spaces = self.rot_mats_to_body_bone_rest_local_spaces.to(self.device)
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

        # print("Input to fish model:")
        # print(f"  global_ori_plus_body_pose: {global_ori_plus_body_pose}")
        # print(f"  all_bone_lengths: {all_bone_lengths}")

        # print(f"all_bone_lengths: {all_bone_lengths.size()}")
        # print(f"global_ori_plus_body_pose: {global_ori_plus_body_pose.size()}")


        # LBS
        if deform:
            verts, global_ori_plus_body_pose_template_space = self.LBS(V, global_ori_plus_body_pose, all_bone_lengths, scale, to_rotmats=pose2rot)
            # global_ori_plus_body_pose_template_space describes the *active* rotation of each bone in the coordinate system where the y-axis
            # is the head-joint-to-first-body-joint direction in the rest pose.
            # However, each bodybone's rest-pose twist axis might not be aligned with the head-joint-to-first-body-joint direction in the rest pose. 
            # We need to convert the bone pose from the mesh local space to the local space of the rest bone head (where the y-axis is the twist axis)
            # in order to align the twist axis of the rotation with the twist axis of the bone in the rest pose.
            global_ori_mat = global_ori_plus_body_pose_template_space[:, 0] # (1, 3, 3) global rotation in mesh local space
            body_bone_poses_rest_bone_spaces = (
                self.rot_mats_to_body_bone_rest_local_spaces.unsqueeze(0) @ global_ori_plus_body_pose_template_space[:, 1:]
            ) # (1, n_body_bones, 3, 3)
            # comment: tail_artic_local = to_local @ pose_world @ tail_rest_world
            # @ is associative, so we can do 
            # tail_artic_local = M @ tail_rest_world
            # where M=(to_local @ pose_world)

            # print("LBS output:")
            # print(f"   body_pose_template_space: {body_pose_template_space}")
            # print(f"   body_bone_poses_rest_bone_spaces: {body_bone_poses_rest_bone_spaces}")

            global_ori_plus_body_pose_mats = torch.cat(
                [global_ori_mat.unsqueeze(1), body_bone_poses_rest_bone_spaces], dim=1
            ) # (1, n_bones, 3, 3)
            # PyTorch3D's matrix_to_quaternion returns quaternions with real part first, as tensor of shape (…, 4).
            global_ori_plus_body_pose_rest_bone_spaces = matrix_to_quaternion(
                global_ori_plus_body_pose_mats.squeeze()
            ).unsqueeze(0).cpu() # (1, n_bones, 4) (w, x, y, z)
        else:
            verts, global_ori_plus_body_pose_template_space = V, None

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = []
        for i in range(verts.shape[0]):
            kpt = torch.matmul(self.vert2kpt, verts[i])
            keypoints.append(kpt)
        keypoints = torch.stack(keypoints)

        # Final output after articulation
        output = {"vertices": verts.cpu(), "keypoints": keypoints.cpu(), "global_ori_plus_body_pose_rest_bone_spaces": global_ori_plus_body_pose_rest_bone_spaces}

        return output
