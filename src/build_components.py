import yaml
import torch
import torch.optim as optim
from src.models.point_gnn import PointGNN
from torch.optim.lr_scheduler import StepLR
from src.criterions.losses import MultiTaskLoss
from src.data_handlers.dataset import GraphKITTI
from src.data_handlers.data_loader import CustomDataLoader, InfiniteDataLoader


def build_dataloaders(cfg):
    """
    Build required dataloaders from config.
    :param cfg: config dict
    :return: dict containing PyTorch data loaders with key options: ["train",
    "val", "test"]
    """

    try:
        root_dir = cfg['root_dir']
        use_rgb = cfg['use_rgb']
        voxel_size = cfg['voxel_size']
        r = cfg['r']
        r0 = cfg['r0']
        max_num_edges = cfg['max_num_edges']
        batch_size = cfg['batch_size']
        train_set = GraphKITTI(root_dir, split='training', use_rgb=use_rgb,
                               voxel_size=voxel_size, r=r, r0=r0,
                               max_num_edges=max_num_edges)
        train_loader = InfiniteDataLoader(train_set, batch_size, shuffle=True)

        test_set = GraphKITTI(root_dir, split='testing', use_rgb=use_rgb,
                              voxel_size=voxel_size, r=r, r0=r0,
                              max_num_edges=max_num_edges)
        test_loader = CustomDataLoader(test_set, batch_size, shuffle=False)

        return {'train': train_loader, 'val': test_loader}

    except KeyError:
        print('Not all required dataset parameters specified in config file.'
              'Exiting...')
        exit()


def build_model(cfg):
    """
    Build model from config dict.
    :param cfg: config dict
    :return: PyTorch nn.Module model
    """

    try:
        kp_dim = cfg['kp_dim']
        state_dim = cfg['state_dim']
        n_classes = cfg['n_classes']
        n_iterations = cfg['n_iterations']
        model = PointGNN(kp_dim, state_dim, n_classes, n_iterations)
        return model
    except KeyError:
        print('Not all required model parameters specified in config file.'
              'Exiting...')
        exit()


def build_criterion(cfg):
    """
    Build loss object from config dict.
    :param cfg: config dict
    :return: PyTorch nn.Module object
    """

    try:
        object_classes = cfg['object_classes']
        lambdas = cfg['lambdas']
        return MultiTaskLoss(object_classes=object_classes, lambdas=lambdas)
    except KeyError:
        print('Not all required loss parameters specified in config file.'
              'Exiting...')
        exit()


def build_lr_scheduler(cfg, optimizer):
    if cfg.get('LR_Scheduler') is None:
        return None
    else:
        try:
            step_size = cfg['step_size']
            gamma = cfg['gamma']
            return StepLR(optimizer, step_size=step_size, gamma=gamma,
                          last_epoch=-1)
        except KeyError:
            print('Not all required scheduler parameters specified in config '
                  'file. Exiting...')
            exit()


def build_metric_trackers(cfg):
    return {'train': MetricTracker(), 'eval': MetricTracker()}


def build_optimizer(cfg, params):
    """
    Build optimizer from config dict.
    :param cfg: config dict
    :param params: network parameters to be optimized
    :return:
    """

    try:
        learning_rate = cfg['learning_rate']
        return optim.SGD(params, lr=learning_rate)
    except KeyError:
        print('Not all required optimizer parameters specified in config file.'
              'Exiting...')
        exit()


if __name__ == '__main__':

    def flatten_cfg(cfg):
        """
        Flatten config dictionary. Removing sub-dicts allows for unified access to
        all parameters. Unique keys are required for all parameters.
        :param cfg: config dictionary with potentially several sub-dicts
        :return: cfg_f: flatten config dict containing no other dicts
        """

        cfg_f = {}

        def recurse(t, key=''):

            # Flatten dictionary recursively
            if isinstance(t, dict):
                for k, v in t.items():
                    recurse(v, k)

            # Add item if no dictionary
            else:
                # Check for ambiguous parameter names
                if key in cfg_f.keys():
                    print(f'Error occurred while flattening config file. Key '
                          f'"{key}" is provided more than once. \n'
                          f'Unique keys are '
                          f'required for all parameters. Exiting...')
                    exit()

                cfg_f[key] = t

        # Start recursion
        recurse(cfg)

        return cfg_f


    # Create config
    with open('../config.yaml', 'r') as f:
        cfg = flatten_cfg(yaml.safe_load(f))

    # Build data loaders
    data_loaders = build_dataloaders(cfg=cfg)

    print(f'Dataset Size: {len(data_loaders["train"].dataset)}')

    # Build network
    device = torch.device('cuda:0')
    model = build_model(cfg=cfg).to(device)

    # Build criterion
    criterion = build_criterion(cfg=cfg)

    # Build optimizer
    optimizer = build_optimizer(cfg=cfg, params=model.parameters())

    # Iterate over dataset
    for batch_id, batch_data in enumerate(data_loaders['train']):

        # Send data items to GPU
        batch_data = {data_key: data_item.to(device) for data_key, data_item in
                      batch_data.items()}

        # Set gradients to zero
        optimizer.zero_grad()

        print(f'Batch ID: {batch_id + 1}')

        # print(f'Batch Input: {batch_data.size()}')
        # print(f'Batch Label: {batch_labels.size()}')

        # Perform forward pass
        cls_pred, loc_pred = model(batch_data)

        # print(f'Batch Output: {batch_output.size()}')
        cls_target = batch_data['labels'][:, :1].long()
        loc_target = batch_data['labels'][:, 1:].float()

        # Compute loss
        loss = criterion(cls_pred, loc_pred, cls_target, loc_target,
                         model.parameters())

        # print(f'Loss: {loss}')

        # Perform backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

