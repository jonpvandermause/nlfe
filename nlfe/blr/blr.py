import numpy as np
import torch
import torch.nn.functional as F


def one_layer_final(x, network):
    """Return final layer of single layer network."""

    x = torch.tensor([x]).float()
    x = torch.tensor([torch.sin(x[0][0]), torch.cos(x[0][0]),
                      torch.sin(x[0][1]), torch.cos(x[0][1])])
    x = F.tanh(network.lin1(x))

    return np.array(x.detach())

def two_layer_final(x, network):
    """Return final layer of two layer network."""

    x = torch.tensor([x]).float()
    x = torch.tensor([torch.sin(x[0][0]), torch.cos(x[0][0]),
                      torch.sin(x[0][1]), torch.cos(x[0][1])])
    x = F.tanh(network.lin1(x))
    x = F.tanh(network.lin2(x))

    return np.array(x.detach())


def three_layer_final(x, network):
    """Return final layer of two layer network."""

    x = torch.tensor([x]).float()
    x = torch.tensor([torch.sin(x[0][0]), torch.cos(x[0][0]),
                      torch.sin(x[0][1]), torch.cos(x[0][1])])
    x = F.tanh(network.lin1(x))
    x = F.tanh(network.lin2(x))
    x = F.tanh(network.lin3(x))

    return np.array(x.detach())


def return_final_layer(x, network):
    """Return the final hidden layer of the two hidden layer network defined
    in nets.py. Input and output are numpy arrays."""

    x = torch.tensor([x]).float()
    x = torch.tensor([torch.sin(x[0][0]), torch.cos(x[0][0]),
                      torch.sin(x[0][1]), torch.cos(x[0][1])])
    x = F.tanh(network.lin1(x))
    x = F.tanh(network.lin2(x))

    return np.array(x.detach())


def get_blr_mean_and_std(test_vector, wmle, bn, an, vn):
    """Compute the mean and standard deviation of the BLR posterior
    predictive with uninformative prior, which takes the form of a student
    t distribution (see p. 39 of Murphy)."""

    mean = np.dot(test_vector, wmle)
    dof = 2 * an  # degrees of freedom in student t
    scale_ratio = bn / an
    epistemic = np.dot(test_vector, np.dot(vn, test_vector.transpose()))
    aleatoric = 1
    scale_param = scale_ratio * (aleatoric + epistemic)
    variance = dof * scale_param / (dof - 2)  # variance of student t
    std = np.sqrt(variance)

    return mean, std, epistemic


def get_blr_matrices(design_matrix, train_labels):
    """Return key matrices and vectors needed to compute the mean and variance
    of the BLR posterior predictive distribution with an uninformative
    prior. Code follows section 7.6.3 of Murphy's book."""

    design_trans = design_matrix.transpose()
    vn = np.linalg.inv(np.dot(design_trans, design_matrix))
    wmle = np.dot(vn, np.dot(design_trans, train_labels))
    N = design_matrix.shape[0]
    D = design_matrix.shape[1]
    an = (N-D)/2
    model_ests = np.dot(design_matrix, wmle)
    diff_vector = train_labels - model_ests
    s_sq = np.dot(diff_vector.transpose(), diff_vector)
    bn = s_sq / 2

    return wmle, bn, an, vn
