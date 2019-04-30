import numpy as np
from atomeye_helper import write_cfg_file

# -----------------------------------------------------------------------------
#                   load coordinate and cell information
# -----------------------------------------------------------------------------

pos_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/\
water_positions.npy'

cell_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/\
cells.npy'

all_positions = np.load(pos_file)
all_cells = np.load(cell_file)

# visualize final protein
protein_pos = all_positions[-1] * 10
protein_cell = all_cells[-1] * 10

# set water species
protein_species = ['H', 'C', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'C',
                   'H', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'H', 'H']

for n in range(372):
    protein_species.append('O')
    protein_species.append('H')
    protein_species.append('H')


# -----------------------------------------------------------------------------
#                              write cfg file
# -----------------------------------------------------------------------------

cfg_file = 'waters.cfg'
write_cfg_file(cfg_file, protein_pos, protein_species, protein_cell)
