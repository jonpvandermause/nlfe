import numpy as np
import matplotlib.pyplot as plt

times_file = \
    '/Users/jonpvandermause/Research/CV/nlfe/nlfe/gromacs_parsers/times.npy'
times = np.load(times_file)
psis = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/dihedrals/psis.npy')
phis = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/dihedrals/phis.npy')

hist, xarray, yarray = np.histogram2d(psis, phis, 100)
prob = hist / 1000001
log_prob = np.log(prob)
free_energy = -2.479 * log_prob

print(xarray)
print(yarray)

# construct training set
skip = 8
train_pts = []
train_labels = []

for m in np.arange(0, 100, skip):
    for n in np.arange(0, 100, skip):
        label = free_energy[m, n]

        if label != np.inf:
            # psi_curr = (xarray[m] + xarray[m+1]) / 2
            # phi_curr = (yarray[n] + yarray[n+1]) / 2

            psi_curr = xarray[m+1]
            phi_curr = yarray[n+1]

            train_pts.append(np.array([psi_curr, phi_curr]))
            train_labels.append(label)

train_pts = np.array(train_pts)
train_labels = np.array(train_labels)

# construct test set
test_pts = []
test_labels = []

for m in np.arange(int(skip/2), 100, skip):
    for n in np.arange(int(skip/2), 100, skip):
        label = free_energy[m, n]
        # psi_curr = (xarray[m] + xarray[m+1]) / 2
        # phi_curr = (yarray[n] + yarray[n+1]) / 2

        psi_curr = xarray[m+1]
        phi_curr = yarray[n+1]

        if label != np.inf:
            test_pts.append(np.array([psi_curr, phi_curr]))
            test_labels.append(label)
        else:
            print(psi_curr)
            print(phi_curr)
            print(label)

test_pts = np.array(test_pts)
test_labels = np.array(test_labels)

# save train and test sets
np.save('train_pts', train_pts)
np.save('train_labels', train_labels)
np.save('test_pts', test_pts)
np.save('test_labels', test_labels)


# plot train and test points
for train_pt, test_pt in zip(train_pts, test_pts):
    plt.plot(train_pt[1], train_pt[0], 'k.')
    plt.plot(test_pt[1], test_pt[0], 'r.')

plt.show()
