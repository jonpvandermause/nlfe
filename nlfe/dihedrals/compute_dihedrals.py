import numpy as np
import math


def get_dihedral_from_coordinates(x1, x2, x3, x4):
    v1 = x2 - x1
    v2 = x3 - x2
    v3 = x4 - x3

    cross1 = np.cross(v1, v2)
    cross2 = np.cross(v2, v3)

    n1 = cross1 / np.linalg.norm(cross1)
    n2 = cross2 / np.linalg.norm(cross2)

    v2_norm = v2 / np.linalg.norm(v2)
    new_ax = np.cross(v2_norm, n1)

    x = np.dot(n2, n1)
    y = np.dot(n2, new_ax)

    dihedral = math.atan2(y, x)

    return dihedral

# print(get_dihedral_from_coordinates(np.array([0, 0, 2]),
#                                     np.array([0, 0, 0]),
#                                     np.array([-1.2, 0, 0]),
#                                     np.array([-1, -1, 0])))


bb_pos = np.load('bb_pos.npy')

phis = np.zeros(bb_pos.shape[0])
psis = np.zeros(bb_pos.shape[0])

for n, pos in enumerate(bb_pos):
    phis[n] = get_dihedral_from_coordinates(pos[0], pos[1], pos[2], pos[3])
    psis[n] = get_dihedral_from_coordinates(pos[1], pos[2], pos[3], pos[4])

np.save('phis', phis)
np.save('psis', psis)
