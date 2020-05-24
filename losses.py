import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss
import torch.nn.functional as F
from model import PointGNN


class FocalLoss(nn.Module):
    # From https://github.com/mbsariyildiz/focal-loss.pytorch/
    # blob/master/focalloss.py
    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target, reduction='mean'):

        # Compute class probability
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        pt = logpt.exp()

        # Compute focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError


class LocalizationLoss(nn.Module):
    def __init__(self):
        super(LocalizationLoss, self).__init__()
        self.loc_loss = SmoothL1Loss(reduction='sum')

    def forward(self, loc_pred, loc_target):

        # Initialize regression loss
        loc_loss = 0

        # Iterate over regression components
        for i in range(loc_pred.size(1)):

            # Compute loss components
            loc_loss += self.loc_loss(loc_pred[i], loc_target[i])

        return loc_loss


class MultiTaskLoss(nn.Module):
    def __init__(self, object_classes, lambdas):
        super(MultiTaskLoss, self).__init__()

        # List of object classes
        self.object_classes = object_classes

        # Set loss weights
        self.lambdas = lambdas

        # Loss components
        self.cls_loss = FocalLoss(gamma=0)
        self.loc_loss = LocalizationLoss()

    def forward(self, cls_pred, loc_pred, cls_target, loc_target, params):

        # Compute classification loss
        cls_loss = self.cls_loss(cls_pred, cls_target)

        # Compute indices of object vertices
        object_inds = torch.nonzero(cls_target.squeeze(1) == torch
                                    .tensor(self.object_classes))\
            .repeat(1, loc_target.size(1))

        # Get object predictions and targets
        object_loc_pred = torch.gather(loc_pred, 0, object_inds)
        object_loc_target = torch.gather(loc_target, 0, object_inds)

        # Compute regression loss
        loc_loss = self.loc_loss(object_loc_pred, object_loc_target)

        # Normalize regression loss w.r.t total number of vertices
        loc_loss /= loc_target.size(0)

        # Compute L1 regularization loss
        reg_loss = 0
        for param in params:
            reg_loss += torch.norm(param, 1)

        # Compute total loss
        lambda1, lambda2, lambda3 = self.lambdas
        multi_task_loss = lambda1 * cls_loss + lambda2 * loc_loss + \
                          lambda3 * reg_loss

        return multi_task_loss


if __name__ == '__main__':

    batch_size = 3
    n_vertices = 1000
    n_classes = 5

    cls_pred = torch.rand((n_vertices, n_classes))
    cls_target = torch.randint(0, n_classes, (n_vertices, 1))
    reg_pred = torch.rand((n_vertices, 7))
    reg_target = torch.rand((n_vertices, 7))
    model = PointGNN(n_classes=5, n_iterations=5, kp_dim=3, state_dim=3)
    named_params = model.parameters()

    criterion = MultiTaskLoss(lambdas=(0.1, 10.0, 5e-7), object_classes=[1])

    loss = criterion(cls_pred, reg_pred, cls_target, reg_target, named_params)
    print(f'Loss: {loss}')