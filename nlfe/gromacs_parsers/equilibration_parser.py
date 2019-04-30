import numpy as np
import matplotlib.pyplot as plt

pe_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/ala_data/potential.xvg'
temp_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/ala_data/temperature.xvg'
dens_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/ala_data/density.xvg'


def energy_parser(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    times = []
    energies = []

    for line in lines:
        if not (line.startswith('#') or line.startswith('@')):
            line_split = line.split()
            time = float(line_split[0])
            energy = float(line_split[1])

            times.append(time)
            energies.append(energy)

    times = np.array(times)
    energies = np.array(energies)

    return times, energies

# parse files
pe_steps, pes = energy_parser(pe_file)
pe_steps = np.array([int(n) for n in pe_steps])

temp_time, temps = energy_parser(temp_file)
dens_time, dens = energy_parser(dens_file)

# make plot
f, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(pe_steps, pes)
ax1.set_xlabel('minimization step')
ax1.set_ylabel('energy (kJ/mol)')
ax1.set_title('energy minimization')

ax2.plot(temp_time, temps)
ax2.set_xlabel('time (ps)')
ax2.set_ylabel('temperature (K)')
ax2.set_title('NVT equilibration')

ax3.plot(dens_time, dens)
ax3.set_xlabel('time (ps)')
ax3.set_ylabel('density (kg/m$^3$)')
ax3.set_title('NPT equilibration')


# # add a, b, c labels
# ax1.text(-40, 2800, '(a)')
# ax1.text(110, 2800, '(b)')
# ax1.text(250, 2800, '(c)')

f.subplots_adjust(hspace=0.8)
plt.show()
