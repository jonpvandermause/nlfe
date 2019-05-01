import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import nets

train_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_pts.npy')
train_labels = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/train_labels.npy')
test_pts = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/test_pts.npy')
test_labels = \
    np.load('/Users/jonpvandermause/Research/CV/nlfe/nlfe/nn/test_labels.npy')


def train_net(net, net_name):
    net.zero_grad()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    loss_list = []
    order = np.arange(len(train_pts))

    for epoch in range(10):  # loop over the dataset multiple times

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

    loss_list = np.array(loss_list)

    torch.save(net, net_name)

    return loss_list

# train single layer nets
for n in range(4):
    initial_net = nets.OneLayer(30)
    net_name = 'saved_models/one_layer_'+str(n)
    loss_list = train_net(initial_net, net_name)
    np.save('saved_models/one_layer_loss_'+str(n), loss_list)


# train two layer nets
for n in range(4):
    initial_net = nets.TwoLayer(20, 10)
    net_name = 'saved_models/two_layer_'+str(n)
    loss_list = train_net(initial_net, net_name)
    np.save('saved_models/two_layer_loss_'+str(n), loss_list)

# train three layer nets
for n in range(4):
    initial_net = nets.ThreeLayer(48, 24, 12)
    net_name = 'saved_models/three_layer_'+str(n)
    loss_list = train_net(initial_net, net_name)
    np.save('saved_models/three_layer_loss_'+str(n), loss_list)
