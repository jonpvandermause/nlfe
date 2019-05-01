import numpy as np

train_diffs = np.load('train_diffs.npy')
train_diffs_ensemble = np.load('train_diffs_ensemble.npy')
test_diffs = np.load('test_diffs.npy')
test_diffs_ensemble = np.load('test_diffs_ensemble.npy')

for m in range(5):
    print('\n--------------- ensemble '+str(m)+' --------------------------')
    for n in range(4):
        print('\nensemble '+str(m)+' network '+str(n) + ' train error')
        print(np.mean(np.abs(train_diffs[:, m, n])))

        print('\nensemble '+str(m)+' network '+str(n) + ' test error')
        print(np.mean(np.abs(test_diffs[:, m, n])))

    print('\nensemble train error')
    print(np.mean(np.abs(train_diffs_ensemble[:, m])))

    print('\nensemble test error')
    print(np.mean(np.abs(test_diffs_ensemble[:, m])))
