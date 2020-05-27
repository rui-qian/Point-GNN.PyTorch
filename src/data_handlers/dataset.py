import os
import torch
import numpy as np
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.transforms import RadiusGraph
import src.data_handlers.dataset_utils as u


class GraphKITTI(Dataset):
    """
    KITTI Dataset for Graph Neural Networks.
    """

    def __init__(self, root_dir, split='training', use_rgb=False,
                 voxel_size=0.8, r=4.0, r0=1.0, max_num_edges=256):
        """
        Dataset for training and testing containing point cloud, calibration
        object and in case of training labels
        :param root_dir: root directory of the dataset
        :param split: training or testing split of the dataset
        :param use_rgb: use RGB information by finding corresponding pixels
        :param voxel_size: side length of cubic voxels for down-sampling
        :param r: radius for graph construction in m
        :param r0: radius for key-point search in m
        :param max_num_edges: maximum number of graph edges per vertex
        """

        # Set parameters
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        # Set number of samples depending on split
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

        # Set radii
        self.r = r
        self.r0 = r0

        # Set maximum number of edges per vertex
        self.max_num_edges = max_num_edges

        # Set paths to camera, LiDAR, calibration and label directories
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
        calib = u.Calibration(calib_filename)

        # Get point cloud
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % index)
        point_cloud_velo = u.load_velo_scan(lidar_filename)

        # Get image
        image_filename = os.path.join(self.image_dir, '%06d.png' % index)
        image = u.get_image(image_filename)

        # Get labels
        label_filename = os.path.join(self.label_dir, '%06d.txt' % index)
        labels = u.read_label(label_filename)

        # Create point cloud object
        point_cloud_ref = u.PointCloud(point_cloud_velo.xyz,
                                       point_cloud_velo.feats)

        # Transform point cloud to camera coordinates
        point_cloud_ref.xyz = calib.project_velo_to_ref(point_cloud_velo.xyz)

        # Down-sample point cloud
        point_cloud_ref = u.downsample_by_average_voxel(
            point_cloud_ref, self.voxel_size)

        # Filter out points outside of camera's FoV
        point_cloud_ref = u.filter_point_cloud(point_cloud_ref,
                                                         image, calib)

        # Add RGB information as additional input features
        if self.use_rgb:
            point_cloud_ref = u.get_rgb_feats(point_cloud_ref, image)

        # Create graph
        vertex_xyz = torch.from_numpy(point_cloud_ref.xyz).float()
        n_vertices = vertex_xyz.size(0)
        data = Data(pos=vertex_xyz)
        graph = RadiusGraph(r=self.r, loop=False,
                            max_num_neighbors=self.max_num_edges,
                            flow='target_to_source')(data)

        # Get raw point cloud in camera coordinates
        # NOTE: Raw point cloud required to compute initial vertex states for
        # down-sampled graph representation
        point_cloud_raw_xyz = calib.project_velo_to_ref(point_cloud_velo.xyz)

        # Get key-point indices for each vertex using KD Trees
        vertex_tree = cKDTree(vertex_xyz.numpy())
        raw_point_cloud_tree = cKDTree(point_cloud_raw_xyz)
        key_point_indices = vertex_tree.query_ball_tree(raw_point_cloud_tree,
                                                        r=self.r0, p=2)

        # Compute initial key-point features
        key_points = None
        key_points_lookup = torch.zeros(n_vertices).type(torch.int64)
        for vertex_id, xyz in enumerate(vertex_xyz.numpy()):

            # Get key-point indices for currently inspected vertex
            vertex_kp_indices = np.array(key_point_indices[vertex_id])

            # Compute relative offsets from vertex
            vertex_kp_offsets = point_cloud_raw_xyz[vertex_kp_indices] \
                                - xyz

            # Retrieve key-point reflectance
            vertex_kp_feats = point_cloud_velo.feats[vertex_kp_indices]

            # Create key-points by concatenating features
            vertex_kps = torch.from_numpy(np.concatenate((vertex_kp_offsets,
                                                         vertex_kp_feats),
                                                         axis=1))
            # Get number of key-points
            n_key_points = vertex_kps.size(0)

            # Construct look-up table for vertex's key-points
            if vertex_id < (n_vertices-1):
                key_points_lookup[vertex_id+1] = key_points_lookup[vertex_id] \
                                                + n_key_points

            # Concatenate key-along for all vertices along first dimension
            if key_points is None:
                key_points = vertex_kps
            else:
                key_points = torch.cat((key_points, vertex_kps), dim=0)

        # Compute vertex labels
        cls_labels, boxes_3d, _, _ = \
            u.get_point_labels(labels, vertex_xyz.numpy(),
                                         expand_factor=(1.1, 1.1, 1.1))
        loc_labels = u.box_encoding(cls_labels, vertex_xyz.numpy(),
                                              boxes_3d)
        vertex_labels = torch.from_numpy(np.concatenate((cls_labels,
                                                         loc_labels), axis=1))

        # Create return dictionary
        rtn_dict = {'graph': graph,
                    'key_points': key_points,
                    'key_points_lookup': key_points_lookup,
                    'labels': vertex_labels}

        return rtn_dict
