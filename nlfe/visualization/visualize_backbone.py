import numpy as np
from atomeye_helper import write_cfg_file

# -----------------------------------------------------------------------------
#                   load coordinate and cell information
# -----------------------------------------------------------------------------

pos_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/\
positions.npy'

cell_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/\
cells.npy'

all_positions = np.load(pos_file)
all_cells = np.load(cell_file)

# visualize final protein
protein_pos = all_positions[-1] * 10
protein_cell = all_cells[-1] * 10
protein_species = ['H', 'C', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'C',
                   'H', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'H', 'H']

print(protein_pos)

# visualize CNCC backbone
CNCC_pos = np.zeros((4, 3))
CNCC_pos[0] = protein_pos[4]
CNCC_pos[1] = protein_pos[6]
CNCC_pos[2] = protein_pos[8]
CNCC_pos[3] = protein_pos[14]

CNCC_species = ['C', 'N', 'C', 'C']

# visualize NCCN backbone
NCCN_pos = np.zeros((4, 3))
NCCN_pos[0] = protein_pos[6]
NCCN_pos[1] = protein_pos[8]
NCCN_pos[2] = protein_pos[14]
NCCN_pos[3] = protein_pos[16]

NCCN_species = ['N', 'C', 'C', 'N']

# -----------------------------------------------------------------------------
#                              write cfg file
# -----------------------------------------------------------------------------

cncc_file = 'cncc.cfg'
write_cfg_file(cncc_file, CNCC_pos, CNCC_species, protein_cell)

nccn_file = 'nccn.cfg'
write_cfg_file(nccn_file, NCCN_pos, NCCN_species, protein_cell)
