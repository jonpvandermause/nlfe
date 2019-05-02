import numpy as np
import matplotlib.pyplot as plt

train_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_pts.npy')
test_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/test_pts.npy')

phi_array = np.load('phi_array.npy')
psi_array = np.load('psi_array.npy')
free_ens = np.load('free_ens.npy')

test_preds = np.load('test_preds.npy')
test_stds = np.load('test_stds.npy')

# -----------------------------------------------------------------------------
#                           make plot
# -----------------------------------------------------------------------------

mod = 0
f, (ax1, ax2, ax3) = plt.subplots(3, 1)

im = ax1.contourf(phi_array, psi_array, test_preds[:, :, mod])
f.colorbar(im, ax=ax1)
ax1.set_xlabel('$\phi$')
ax1.set_ylabel('$\psi$')
ax1.set_title('mean')

im = ax2.contourf(phi_array, psi_array,
                  np.abs(free_ens - test_preds[:, :, mod]))
# plot train and test points
for train_pt, test_pt in zip(train_pts, test_pts):
    ax2.plot(train_pt[1], train_pt[0], 'k.', markersize=1)
    ax2.plot(test_pt[1], test_pt[0], color='gray', marker='.', markersize=1)

f.colorbar(im, ax=ax2)
ax2.set_xlabel('$\phi$')
ax2.set_ylabel('$\psi$')
ax2.set_title('error')


im = ax3.contourf(phi_array, psi_array, test_stds[:, :, mod])
# plot train and test points
for train_pt, test_pt in zip(train_pts, test_pts):
    ax3.plot(train_pt[1], train_pt[0], 'k.', markersize=1)
    ax3.plot(test_pt[1], test_pt[0], color='gray', marker='.', markersize=1)

f.colorbar(im, ax=ax3)
ax3.set_xlabel('$\phi$')
ax3.set_ylabel('$\psi$')
ax3.set_title('uncertainty')

f.set_size_inches(3, 6)
f.subplots_adjust(hspace = 0.7)
f.savefig('/Users/jonpvandermause/Research/CV/nlfe/nlfe/figures/ens_err.pdf',
          format='pdf',bbox_inches='tight')

# plt.show()
