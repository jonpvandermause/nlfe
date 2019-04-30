import matplotlib.pyplot as plt
import numpy as np

times_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/times.npy'
times = np.load(times_file)
psis = np.load('psis.npy')
phis = np.load('phis.npy')

# plt.plot(times, psis, '.', markersize=1)

# plt.hist(psis, 1000)
# plt.show()

# plt.hist(phis, 1000)
# plt.show()

hist, xarray, yarray = np.histogram2d(psis, phis, 100)
print(hist.shape)
print(np.sum(hist))

prob = hist / 1000001
# prob += 1e-8
log_prob = np.log(prob)
free_energy = -2.479 * log_prob

plt.contourf(xarray[1:], yarray[1:], free_energy)
plt.colorbar()
# plt.contour(xarray[1:], yarray[1:], log_prob)
plt.show()

# print(xarray.shape)
# print(yarray.shape)
# print(hist.shape)


# plt.hist2d(phis, psis, 1000)
# plt.show()
