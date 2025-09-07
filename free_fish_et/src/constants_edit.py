"""
parameters and constrains, modified from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""

import torch

"""
Body_pose angle limit
we minus index by 1 because we exclude root pose as it is modeled as global orient
"""
max_lim = [0.] * (num_bone*3)
min_lim = [0.] * (num_bone*3)

# for i in range(num_bone):
#     max_lim[i * 3: (i + 1) * 3] = 0, 1.2, 0.02
#     min_lim[i * 3: (i + 1) * 3] = 0, -1.2, -0.02
#
# for i in range(num_bone - 4, num_bone):
#     max_lim[i * 3: (i + 1) * 3] = 0, 1.5, 0.03
#     min_lim[i * 3: (i + 1) * 3] = 0, -1.5, -0.03

for i in range(num_bone):
    max_lim[i * 3: (i + 1) * 3] = 0., -0.05, 0.
    min_lim[i * 3: (i + 1) * 3] = -0., -0.05, 0.

for i in range(num_bone - 3, num_bone):
    max_lim[i * 3: (i + 1) * 3] = 0.0, -0.05, 0.
    min_lim[i * 3: (i + 1) * 3] = -0.0, -0.05, 0.

for i in range(num_bone - 1, num_bone):
    max_lim[i * 3: (i + 1) * 3] = 0.0, -0.07, 0.
    min_lim[i * 3: (i + 1) * 3] = -0.0, -0.07, 0.

"""
Body bone length limit
"""
# max_bone = [2.4] * (num_bone)
max_bone = [2.3] * (num_bone)
min_bone = [1.0] * (num_bone)

# max_bone[0] = 2.7
# max_bone[1] = 2.7
# max_bone[2] = 2.7
# max_bone[3] = 1.7

# min_bone[0] = 0.3
# min_bone[1] = 0.5
# min_bone[2] = 0.4
# min_bone[3] = 0.4
