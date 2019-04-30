import numpy as np

times_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/times.npy'
pos_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/\
positions.npy'
cell_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/\
cells.npy'

times = np.load(times_file)
positions = np.load(pos_file)
cells = np.load(cell_file)

bb_inds = np.array([4, 6, 8, 14, 16])

# compute unwrapped backbone positions
bb_pos = np.zeros((len(positions), 5, 3))

for n, (pos, cell) in enumerate(zip(positions, cells)):
    for p in range(3):
        for m, bb_ind in enumerate(bb_inds):
            # first atom is reference
            if m == 0:
                ref_coord = pos[bb_ind, p]
                bb_pos[n, m, p] = ref_coord

            # translate to minimize distance from first atom
            else:
                coord = pos[bb_ind, p]
                save_coord = coord
                diff0 = np.abs(coord - ref_coord)

                coord1 = coord + cell[p, p]
                diff1 = np.abs(coord1 - ref_coord)
                if diff1 < diff0:
                    save_coord = coord1

                coord2 = coord - cell[p, p]
                diff2 = np.abs(coord2 - ref_coord)
                if (diff2 < diff1) and (diff2 < diff0):
                    save_coord = coord2

                bb_pos[n, m, p] = save_coord

np.save('bb_pos', bb_pos)
