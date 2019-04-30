import numpy as np

# -----------------------------------------------------------------------------
#                           parse position coordinates
# -----------------------------------------------------------------------------

filename = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/ala_data/water_pos_2.xvg'

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
positions = positions.reshape(1001, -1, 3)

np.save('water_times', times)
np.save('water_positions', positions)
