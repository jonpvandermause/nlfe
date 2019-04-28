import numpy as np

filename = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/ala_data/solv_pos.xvg'

with open(filename, 'r') as f:
    lines = f.readlines()

times = []
positions = []

for index, line in enumerate(lines):
    if not (line.startswith('#') or line.startswith('@')):
        line_split = line.split()
        time = float(line_split[0])
        position = np.array([float(n) for n in line_split[1:]])

        times.append(time)
        positions.append(position)

times = np.array(times)
positions = np.array(positions)

np.save('times', times)
np.save('positions', positions)
