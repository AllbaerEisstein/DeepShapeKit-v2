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

import json
import torch
from .LBS_edit import LBS
from pytorch3d.transforms import matrix_to_quaternion


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
        # translate to head-local coords (head joint -> origin)
        translation_head_to_origin = self.J[0, 0].unsqueeze(0)
        self.J = self.J - translation_head_to_origin

        self.V = torch.tensor(template["V"]).unsqueeze(0) # (1,V,3)
        # apply the same translation to the vertices as we did to the joints in order to keep them in the same local space
        self.V = self.V - translation_head_to_origin

        # when calculating the swing-twist of each bone, we need to express each articulated bone tail 
        # in the local space of the rest bone head because the twist axis is required to be one of x,y, or z.
        template_to_bone_spaces = []
        for bone_name in template["bone_order"]:
            bone_to_template = torch.tensor(template["bone_names_tree"][bone_name]["rest_rot_world"])
            template_to_bone = bone_to_template.T
            template_to_bone_spaces.append(template_to_bone)
        self.template_to_bone_spaces = torch.stack(template_to_bone_spaces, dim=0).unsqueeze(0) # (1, n_bones, 3, 3)

        # # scaling, unit conversion
        # self.V = self.V #* 0.01
        # self.J = self.J #* 0.01

        self.LBS = LBS(self.J, self.parent_indices, self.weights, self.virtual_bone_mask)

        # CLAUDE FIX: verify at load time that skinning the template with a zero pose returns the
        # template itself. This catches an inconsistent template or a regression in the skinning
        # transform immediately, instead of letting the optimizer silently fit a distorted mesh.
        self.LBS.assert_rest_pose_is_identity(self.V)

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
        self.template_to_bone_spaces = self.template_to_bone_spaces.to(self.device)
        self.device_active = True


    def __call__(self, global_ori, body_pose, body_bone_length, scale=torch.tensor(1), pose2rot=True, deform=True, output_device=None):
        """
        Args:
            global_ori (BS, 3): BS (batch-size) different exponential map representations of global rotation
            body_pose (BS, bbn*3): exponential map representation of body pose (exclude root joint orient) -> orientation of bbn body bones
            body_bone_length (BS, bn): bone lengths for bbn body bones
            scale (BS, 1): scale factor
            pose2rot: if True, convert exponential map to rotation matrix inside LBS
            deform (bool): toggle if linear blend skinning should be applied. If not, just return rest-pose vertices and keypoints in local (model) space.
            output_device: device the returned tensors should live on. Defaults to the model's own
                device, which avoids a host round-trip in the optimizer's inner loop. Pass
                torch.device("cpu") (or "cpu") to get CPU tensors back.
        Returns:
            dict with (at least) the following entries. Callers should index by key rather than
            unpack positionally, so that additional outputs can be added without breaking them:
            keypoints (BS, kn, 3): coordinates of the kn keypoints after LBS in local (model) space
            vertices (BS, vn, 3): coordinates of the vn vertices after LBS in local (model) space
            global_ori_plus_body_pose_rest_bone_spaces (BS, n_bones, 4): each bone's own rotation,
                expressed as a quaternion (w, x, y, z) in that bone's rest frame. Index 0 is the
                global orientation; indices 1.. are the body bones in bone order.
        """
        assert body_bone_length.size(1) == self.n_body_bones, f"body_bone_length must have size (batch_size, n_bones_total-1) but has size {body_bone_length.size()} instead of ({global_ori.size(0), self.n_body_bones})"
        assert global_ori.size(1) == 3, f"global_ori must have size (batch_size, 3) but has size {global_ori.size()} instead of ({global_ori.size(0), 3}) and must supply world space orientation in exponential map representation"
        assert body_pose.size(1) == self.n_body_bones*3, f"body_pose must have size (batch_size, (n_bones_total-1)*3) but has size {body_pose.size()} instead of ({global_ori.size(0), self.n_body_bones*3})"
        
        # CLAUDE FIX: `scale` is moved to the model device alongside the other inputs. It used to be
        # left untouched, which happened to work only because it is a 0-dim CPU tensor (PyTorch
        # promotes those); a shaped CPU scale tensor would have raised a device mismatch.
        if not all(
            self.device == attr.device
            for attr in [self.faces, global_ori, body_pose, body_bone_length]
        ):
            global_ori = global_ori.to(self.device)
            body_pose = body_pose.to(self.device)
            body_bone_length = body_bone_length.to(self.device)
        if torch.is_tensor(scale) and scale.device != self.device:
            scale = scale.to(self.device)

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
        global_ori_plus_body_pose_rest_bone_spaces = None
        if deform:
            # CLAUDE FIX: LBS now returns each bone's own (local) rotation rather than the
            # accumulated chain rotation, which is what the per-joint swing-twist priors are
            # defined against. Index 0 is the global orientation, so the root bone's priors are
            # now enforceable just like every other bone's.
            verts, local_bone_rotations_template_space = self.LBS(V, global_ori_plus_body_pose, all_bone_lengths, scale, to_rotmats=pose2rot)
            # transform each bone ori from template space to its own rest space
            local_bone_rotations_rest_bone_space = (
                self.template_to_bone_spaces @ local_bone_rotations_template_space @ self.template_to_bone_spaces.transpose(-1,-2)
            ) # (BS, n_bones, 3, 3)

            # PyTorch3D's matrix_to_quaternion returns quaternions with real part first, as tensor of shape (…, 4).
            # CLAUDE FIX: matrix_to_quaternion is applied to the (BS, n_bones, 3, 3) tensor directly.
            # The previous `.squeeze()` collapsed *every* size-1 dimension, so it only produced the
            # right shape for a batch size of exactly 1. It also no longer moves the result to the
            # host: keeping it on the model device avoids a device synchronization on every
            # iteration of every optimization stage.
            global_ori_plus_body_pose_rest_bone_spaces = matrix_to_quaternion(
                local_bone_rotations_rest_bone_space
            ) # (BS, n_bones, 4) (w, x, y, z)
        else:
            verts = V

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = torch.matmul(self.vert2kpt.unsqueeze(0), verts)

        # Final output after articulation.
        # CLAUDE FIX: outputs stay on the model device by default. Callers that want host tensors
        # ask for them explicitly via `output_device`, so the return contract stays a dict and can
        # grow new entries without changing any call site.
        output = {
            "vertices": verts,
            "keypoints": keypoints,
            "global_ori_plus_body_pose_rest_bone_spaces": global_ori_plus_body_pose_rest_bone_spaces,
        }
        if output_device is not None:
            target = torch.device(output_device)
            output = {
                key: (value.to(target) if torch.is_tensor(value) else value)
                for key, value in output.items()
            }

        return output