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

# plt.plot(times, psis, '.', markersize=0.1)
# plt.show()

# plt.plot(times, phis, '.', markersize=0.1)
# plt.show()

plt.contourf(xarray[1:], yarray[1:], free_energy)
plt.tick_params(labelsize=20)
# plt.colorbar()
plt.xlabel('$\phi$ (radians)', fontsize=20)
plt.ylabel('$\psi$ (radians)', fontsize=20)

# plot train and test sets
train_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_pts.npy')
test_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/test_pts.npy')

# plot train and test points
for train_pt, test_pt in zip(train_pts, test_pts):
    plt.plot(train_pt[1], train_pt[0], 'k.')
    plt.plot(test_pt[1], test_pt[0], 'r.')

plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

# plt.title('[kJ/mol]', fontsize=20)

# plt.contour(xarray[1:], yarray[1:], log_prob)
plt.show()

# print(xarray.shape)
# print(yarray.shape)
# print(hist.shape)


# plt.hist2d(phis, psis, 1000)
# plt.show()
