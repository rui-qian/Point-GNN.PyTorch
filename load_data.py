from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
import torch
import os
import cv2
import kitti_utils
import time
from torch_geometric.transforms import RadiusGraph


###################
# dataset classes #
###################

class PointCloud:
    def __init__(self, xyz, feats=None):
        self.xyz = xyz
        self.feats = feats

    def __getitem__(self, item):
        if self.feats is not None:
            return self.xyz[item], self.feats[item]
        else:
            return self.xyz, None


class PointCloudDataset(Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, root_dir, split='training', device=torch.device('cpu'),
                 use_rgb=True, voxel_size=0.8):
        """
        Dataset for training and testing containing point cloud, calibration
        object and in case of training labels
        :param root_dir: root directory of the dataset
        :param split: training or testing split of the dataset
        :param device: device on which dataset will be used
        :param use_rgb: use RGB information by finding corresponding pixels
        :param voxel_size: side length of cubic voxels for down-sampling
        """

        self.device = device
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 6481
        elif split == 'testing':
            self.num_samples = 1000
        else:
            print('Unknown split: %s' % split)
            exit(-1)

        # Use RGB information as input features
        self.use_rgb = use_rgb

        # Set voxel size
        self.voxel_size = voxel_size

        # paths to camera, lidar, calibration and label directories
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.image_dir = os.path.join(self.split_dir, 'image_2')

    def __len__(self):
        # Denotes the total number of samples
        return self.num_samples

    def __getitem__(self, index):

        # Get calibration
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % index)
        calib = kitti_utils.Calibration(calib_filename)

        # Get point cloud
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % index)
        point_cloud_velo = kitti_utils.load_velo_scan(lidar_filename)

        # Get image
        image_filename = os.path.join(self.image_dir, '%06d.png' % index)
        image = kitti_utils.get_image(image_filename)

        # Get labels
        label_filename = os.path.join(self.label_dir, '%06d.txt' % index)
        labels = kitti_utils.read_label(label_filename)

        # Create point cloud object
        point_cloud_ref = PointCloud(point_cloud_velo.xyz,
                                     point_cloud_velo.feats)

        # Transform point cloud to camera coordinates
        point_cloud_ref.xyz = calib.project_velo_to_ref(point_cloud_velo.xyz)

        # Down-sample point cloud
        point_cloud_ref = kitti_utils.downsample_by_average_voxel(
            point_cloud_ref, self.voxel_size)

        # Filter out points outside of camera's FoV
        point_cloud_ref = kitti_utils.filter_point_cloud(point_cloud_ref,
                                                         image, calib)

        # Add RGB information as additional input features
        if self.use_rgb:
            point_cloud_ref = kitti_utils.get_rgb_feats(point_cloud_ref, image)

        # Create graph
        vertex_xyz = torch.from_numpy(point_cloud_ref.xyz).float()
        data = Data(pos=vertex_xyz)
        data = RadiusGraph(r=0.8, loop=True, max_num_neighbors=256,
                           flow='target_to_source')(data)

        # Get raw area points
        vertex_key_points = []

        # Get raw point cloud
        point_cloud_raw_xyz = calib.project_velo_to_ref(point_cloud_velo.xyz)

        # Set radius for key point aggregation
        r0 = 1.0

        # Get key-points for each vertex
        for vertex_id, xyz in enumerate(vertex_xyz.numpy()):
            key_point_inds = ((np.sqrt(np.sum((point_cloud_raw_xyz-xyz)**2,
                                              axis=1)) < r0).nonzero()[0])\
                .astype(np.int64)
            key_point_offsets = point_cloud_raw_xyz[key_point_inds] - xyz
            key_point_feats = point_cloud_velo.feats[key_point_inds]
            key_points = torch.from_numpy(np.concatenate((key_point_offsets,
                                                         key_point_feats),
                                                        axis=1))
            vertex_key_points.append(key_points)

        # Create return dictionary
        rtn_dict = {'data': data, 'key_points': vertex_key_points}
        return rtn_dict


if __name__ == '__main__':

    # create data loader
    root_dir = '/globalwork/datasets/KITTI_object/'
    batch_size = 1
    device = torch.device('cpu')
    dataset = PointCloudDataset(root_dir)
    from model import PointGNN

    net = PointGNN(state_dim=300, n_classes=2, n_iterations=1)

    for batch_id, batch_data in enumerate(dataset):
        cls_pred, reg_pred = net(batch_data)
        print(cls_pred.size())
        print(reg_pred.size())
        exit()
