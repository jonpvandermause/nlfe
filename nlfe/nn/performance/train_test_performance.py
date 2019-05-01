import torch
from nets import OneLayer, TwoLayer, ThreeLayer
import numpy as np

# -----------------------------------------------------------------------------
#                           load train and test sets
# -----------------------------------------------------------------------------

train_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_pts.npy')
train_labels = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_labels.npy')
test_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/test_pts.npy')
test_labels = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/test_labels.npy')

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


# -----------------------------------------------------------------------------
#                          compute train errors
# -----------------------------------------------------------------------------

train_mses = np.zeros((len(network_files), 4))
train_mses_ensemble = np.zeros(len(network_files))
diffs = np.zeros((train_labels.shape[0], len(network_files), 4))
ensemble_diffs = np.zeros((train_labels.shape[0], len(network_files)))

for p, (train_pt, train_label) in enumerate(zip(train_pts, train_labels)):
    for m, ensemble in enumerate(network_files):
        pred_sum = 0
        for n, nfile in enumerate(ensemble):
            net_curr = networks[m][n]

            model_prediction = \
                net_curr(torch.tensor([train_pt]).float()).item()
            pred_sum += model_prediction
            diff = train_label - model_prediction
            diffs[p, m, n] = diff

        ensemble_pred = pred_sum / 4
        ens_diff = train_label - ensemble_pred
        ensemble_diffs[p, m] = ens_diff

np.save('train_diffs', diffs)
np.save('train_diffs_ensemble', ensemble_diffs)


# # ---------------------------------------------------------------------------
# #                          compute test errors
# # ---------------------------------------------------------------------------

test_mses = np.zeros((len(network_files), 4))
test_mses_ensemble = np.zeros(len(network_files))
test_diffs = np.zeros((test_labels.shape[0], len(network_files), 4))
test_ensemble_diffs = np.zeros((test_labels.shape[0], len(network_files)))

for p, (test_pt, test_label) in enumerate(zip(test_pts, test_labels)):
    for m, ensemble in enumerate(network_files):
        pred_sum = 0
        for n, nfile in enumerate(ensemble):
            net_curr = networks[m][n]

            model_prediction = \
                net_curr(torch.tensor([test_pt]).float()).item()
            pred_sum += model_prediction
            diff = test_label - model_prediction
            test_diffs[p, m, n] = diff

        ensemble_pred = pred_sum / 4
        ens_diff = test_label - ensemble_pred
        test_ensemble_diffs[p, m] = ens_diff

print(test_diffs[1])
print(test_ensemble_diffs[15])
np.save('test_diffs', test_diffs)
np.save('test_diffs_ensemble', test_ensemble_diffs)
