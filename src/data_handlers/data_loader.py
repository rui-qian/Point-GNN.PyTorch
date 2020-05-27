import torch
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from src.data_handlers.dataset import GraphKITTI


def build_dataloaders(root_dir, batch_size, use_rgb, voxel_size, r, r0,
                      max_num_edges):
    """
    Build data loaders for training and testing. Use infinite data loader for
    training.
    """

    train_set = GraphKITTI(root_dir, split='training', use_rgb=use_rgb,
                           voxel_size=voxel_size, r=r, r0=r0,
                           max_num_edges=max_num_edges)
    train_loader = CustomDataLoader(train_set, batch_size, shuffle=True)

    test_set = GraphKITTI(root_dir, split='testing', use_rgb=use_rgb,
                          voxel_size=voxel_size, r=r, r0=r0,
                          max_num_edges=max_num_edges)
    test_loader = CustomDataLoader(test_set, batch_size, shuffle=False)

    return train_loader, test_loader


#####################
# Custom DataLoader #
#####################

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, **kwargs):
        """
        Create custom DataLoader object to perform collation of data objects
        and list of key-point arrays.
        """

        def collate(batch):
            """
            Collate batch data. Use standard PyTorch Geometric batching for
            graphs. Stack key-point arrays and look-up tables along first
            dimension.
            """

            # Get list of graph objects (PyT Geometric Data) from items' dict
            list_of_graphs = [data_dict['graph'] for data_dict in batch]

            # Use PyTorch Geometric batching
            batch_graphs = Batch.from_data_list(list_of_graphs)

            # Get lists containing key-point and lookup arrays
            list_of_key_points = [data_dict['key_points'] for data_dict
                                  in batch]
            list_of_lookups = [data_dict['key_points_lookup'] for data_dict
                               in batch]

            # Compute lookup offsets for batch samples
            # NOTE: Each lookup array contains start and end key-point indices
            # for each vertex. Index offset required when stacking key-points
            # along first dimension.
            lookup_offsets = np.cumsum([0] + [kps.size(0) for kps in
                                              list_of_key_points[:-1]])

            # Prepare lookups for stacking by adding the accumulated index
            # offsets
            list_of_shifted_lookups = [lookup + lookup_offsets[lookup_id] for
                                       lookup_id, lookup in
                                       enumerate(list_of_lookups)]

            # Concatenate batch key-points and lookups along first dimensions
            batch_key_points_lookup = torch.cat(list_of_shifted_lookups, dim=0)
            batch_key_points = torch.cat(list_of_key_points, dim=0)

            # Concatenate labels along first dimensions
            list_of_labels = [data_dict['labels'] for data_dict in batch]
            labels = torch.cat(list_of_labels, dim=0)

            return {'graphs': batch_graphs,
                    'key_points': batch_key_points,
                    'key_points_lookup': batch_key_points_lookup,
                    'labels': labels}

        # Create PyTorch DataLoader with custom collate function
        super(CustomDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=lambda batch: collate(batch),
                             **kwargs)


#######################
# Infinite DataLoader #
#######################


class InfiniteDataLoader(CustomDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)

        return batch
