import numpy as np
import torch
import torch.nn.functional as F
from blr import return_final_layer, get_blr_matrices, get_blr_mean_and_std, \
    one_layer_final, two_layer_final, three_layer_final
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
#                          load CV values over FES
# -----------------------------------------------------------------------------

phi_array = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/performance/'
            'phi_array.npy')
psi_array = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/performance/'
            'psi_array.npy')
free_energy = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/performance/'
            'free_ens.npy')


# -----------------------------------------------------------------------------
#                                   load network
# -----------------------------------------------------------------------------

net_file = '../nn/saved_models/three_layer_2'
network = torch.load(net_file)


# -----------------------------------------------------------------------------
#                       construct design matrix and training labels
# -----------------------------------------------------------------------------

train_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_pts.npy')
train_labels = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_labels.npy')

layer_size = len(network.lin4.weight[0])
no_train_pts = len(train_pts)
design_matrix = np.zeros((no_train_pts, layer_size))

for n, train_pt in enumerate(train_pts):
    final_layer_curr = three_layer_final(train_pt, network)
    design_matrix[n] = final_layer_curr

wmle, bn, an, vn = get_blr_matrices(design_matrix, train_labels)


# ---------------------------------------------------------------------------
#                record blr predictions over CV values
# ---------------------------------------------------------------------------

test_preds = np.zeros(phi_array.shape)
test_stds = np.zeros(phi_array.shape)
test_epistemic = np.zeros(phi_array.shape)

for p in range(phi_array.shape[0]):
    for q in range(phi_array.shape[0]):
        psi_curr = psi_array[p, q]
        phi_curr = phi_array[p, q]
        dihedrals = np.array([psi_curr, phi_curr])
        test_input = three_layer_final(dihedrals, network)

        mean, std, epistemic = \
            get_blr_mean_and_std(test_input, wmle, bn, an, vn)

        test_preds[p, q] = mean
        test_stds[p, q] = std
        test_epistemic[p, q] = epistemic

# ---------------------------------------------------------------------------
#                              plot blr predictions
# ---------------------------------------------------------------------------

free_ens = free_energy
train_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_pts.npy')
test_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/test_pts.npy')

f, ax3 = plt.subplots(1, 1)

# im = ax1.contourf(phi_array, psi_array, test_preds)
# f.colorbar(im, ax=ax1)
# ax1.set_xlabel('$\phi$')
# ax1.set_ylabel('$\psi$')
# ax1.set_title('mean')

# im = ax2.contourf(phi_array, psi_array,
#                   np.abs(free_ens - test_preds))
# # plot train and test points
# for train_pt, test_pt in zip(train_pts, test_pts):
#     ax2.plot(train_pt[1], train_pt[0], 'k.', markersize=1)
#     ax2.plot(test_pt[1], test_pt[0], color='gray', marker='.', markersize=1)

# f.colorbar(im, ax=ax2)
# ax2.set_xlabel('$\phi$')
# ax2.set_ylabel('$\psi$')
# ax2.set_title('error')


im = ax3.contourf(phi_array, psi_array, test_stds)
# plot train and test points
for train_pt, test_pt in zip(train_pts, test_pts):
    ax3.plot(train_pt[1], train_pt[0], 'k.', markersize=1)
    ax3.plot(test_pt[1], test_pt[0], color='gray', marker='.', markersize=1)

f.colorbar(im, ax=ax3)
ax3.set_xlabel('$\phi$')
ax3.set_ylabel('$\psi$')
ax3.set_title('uncertainty')

# f.set_size_inches(3, 6)
# f.subplots_adjust(hspace = 0.7)
f.savefig('/Users/jonpvandermause/Research/CV/nlfe/nlfe/figures/three_err.pdf',
          format='pdf',bbox_inches='tight')

plt.show()
