import numpy as np

# -----------------------------------------------------------------------------
#                           parse position coordinates
# -----------------------------------------------------------------------------

filename = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/ala_data/solv_pos.xvg'

with open(filename, 'r') as f:
    lines = f.readlines()

frame_count = 0
times = []
positions = []

for index, line in enumerate(lines):
    if not (line.startswith('#') or line.startswith('@')):
        line_split = line.split()
        time = float(line_split[0])
        position = np.array([float(n) for n in line_split[1:]])

        times.append(time)
        positions.append(position)

        frame_count += 1

times = np.array(times)
positions = np.array(positions)
positions = positions.reshape(1000001, 22, 3)

np.save('times', times)
np.save('positions', positions)

# -----------------------------------------------------------------------------
#                           parse box information
# -----------------------------------------------------------------------------

filename = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/ala_data/solv_box.xvg'

with open(filename, 'r') as f:
    lines = f.readlines()

frame_count = 0
times = []
cells = []

for index, line in enumerate(lines):
    if not (line.startswith('#') or line.startswith('@')):
        line_split = line.split()
        time = float(line_split[0])

        cell = np.eye(3)
        cell[0, 0] = float(line_split[1])
        cell[1, 1] = float(line_split[2])
        cell[2, 2] = float(line_split[3])

        times.append(time)
        cells.append(cell)

times = np.array(times)
cells = np.array(cells)

np.save('cells', cells)
