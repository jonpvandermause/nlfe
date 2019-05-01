import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import nets


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

# 48, 24, 12
# 20, 10
net = nets.OneLayer(30)
net.zero_grad()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


# -----------------------------------------------------------------------------
#                               train network
# -----------------------------------------------------------------------------


loss_list = []
order = np.arange(len(train_pts))

for epoch in range(10000):  # loop over the dataset multiple times

    running_loss = 0.0
    np.random.shuffle(order)

    for i in range(len(train_pts)):
        # get the inputs
        inputs = torch.tensor([train_pts[order[i]]]).float()
        labels = torch.tensor([[train_labels[order[i]]]]).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print(running_loss)
    loss_list.append(running_loss)
print('Finished Training')

torch.save(net, 'net_test')
