import numpy as np
import matplotlib.pyplot as plt

train_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_pts.npy')
test_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/test_pts.npy')

phi_array = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/performance/'
            'phi_array.npy')
psi_array = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/performance/'
            'psi_array.npy')
free_ens = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/performance/'
            'free_ens.npy')

test_preds = np.load('blr_preds.npy')
test_stds = np.load('blr_epistemic.npy')
# test_stds = np.load('blr_stds.npy')

f, (ax1, ax2, ax3) = plt.subplots(1, 3)

im = ax1.contourf(phi_array, psi_array, test_preds, 100)
f.colorbar(im, ax=ax1)

im = ax2.contourf(phi_array, psi_array, np.abs(free_ens - test_preds), 600,)
                #   vmax=5)
# plot train and test points
for train_pt, test_pt in zip(train_pts, test_pts):
    ax2.plot(train_pt[1], train_pt[0], 'k.')

f.colorbar(im, ax=ax2)


im = ax3.contourf(phi_array, psi_array, test_stds, 100)
# plot train and test points
for train_pt, test_pt in zip(train_pts, test_pts):
    ax3.plot(train_pt[1], train_pt[0], 'k.')

f.colorbar(im, ax=ax3)

plt.show()
