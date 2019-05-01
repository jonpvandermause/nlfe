import torch
from nets import OneLayer, TwoLayer, ThreeLayer
import numpy as np

# -----------------------------------------------------------------------------
#                        `` define CV values over FES
# -----------------------------------------------------------------------------

psis = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/dihedrals/psis.npy')
phis = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/dihedrals/phis.npy')

hist, xarray, yarray = np.histogram2d(psis, phis, 100)
prob = hist / 1000001
log_prob = np.log(prob)
free_energy = -2.479 * log_prob

# construct test angles
phi_array = np.zeros((len(xarray)-1, len(yarray)-1))
psi_array = np.zeros((len(xarray)-1, len(yarray)-1))
free_ens = np.zeros((len(xarray)-1, len(yarray)-1))


for m in range(len(xarray)-1):
    for n in range(len(yarray)-1):
        label = free_energy[m, n]
        psi_curr = xarray[m+1]
        phi_curr = yarray[n+1]

        phi_array[m, n] = phi_curr
        psi_array[m, n] = psi_curr
        free_ens[m, n] = label

# -----------------------------------------------------------------------------
#                           load neural networks
# -----------------------------------------------------------------------------

network_files = [['../saved_models/one_layer_0',
                  '../saved_models/one_layer_1',
                  '../saved_models/one_layer_2',
                  '../saved_models/one_layer_3'],
                 ['../saved_models/two_layer_0',
                  '../saved_models/two_layer_1',
                  '../saved_models/two_layer_2',
                  '../saved_models/two_layer_3'],
                 ['../saved_models/three_layer_0',
                  '../saved_models/three_layer_1',
                  '../saved_models/three_layer_2',
                  '../saved_models/three_layer_3'],
                 ['../saved_models/one_layer_10_0',
                  '../saved_models/one_layer_10_1',
                  '../saved_models/one_layer_10_2',
                  '../saved_models/one_layer_10_3'],
                 ['../saved_models/one_layer_20_0',
                  '../saved_models/one_layer_20_1',
                  '../saved_models/one_layer_20_2',
                  '../saved_models/one_layer_20_3']]

networks = [[], [], [], [], []]
for n, ensemble in enumerate(network_files):
    for nfile in ensemble:
        networks[n].append(torch.load(nfile))

# ---------------------------------------------------------------------------
#                record network predictions over CV values
# ---------------------------------------------------------------------------

test_preds = np.zeros((len(xarray)-1, len(yarray)-1, len(network_files)))
test_stds = np.zeros((len(xarray)-1, len(yarray)-1, len(network_files)))

for p in range(len(xarray)-1):
    for q in range(len(yarray)-1):
        psi_curr = psi_array[p, q]
        phi_curr = phi_array[p, q]

        for m, ensemble in enumerate(network_files):
            mod_preds = np.zeros(4)

            for n, nfile in enumerate(ensemble):
                net_curr = networks[m][n]

                model_prediction = \
                    net_curr(torch.tensor([[psi_curr, phi_curr]]).float()).\
                    item()
                mod_preds[n] = model_prediction

            mean_pred = np.mean(mod_preds)
            std_val = np.std(mod_preds)

            test_preds[p, q, m] = mean_pred
            test_stds[p, q, m] = std_val

np.save('phi_array', phi_array)
np.save('psi_array', psi_array)
np.save('free_ens', free_ens)
np.save('test_preds', test_preds)
np.save('test_stds', test_stds)
