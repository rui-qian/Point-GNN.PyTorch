import cv2
import copy
from load_data import PointCloud
from sklearn.neighbors import NearestNeighbors


###################
# 3D Label Object #
###################


class Object3D(object):
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded,
        # 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.height = data[8]  # box height
        self.width = data[9]  # box width
        self.length = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])
        # location (x,y,z) in camera coord.
        self.ry = data[14]
        # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.height, self.width, self.length))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


######################
# Calibration Object #
######################


class Calibration(object):
    """
    Calibration matrices and utils

    ------------------
    coordinate systems
    ------------------
    3d XYZ in <label>.txt are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in Velodyne coord.

    image2 coord:
     ----> x-axis (u)
    |
    |
    v y-axis (v)

    velodyne coord:
    front x, left y, up z

    rect/ref camera coord:
    right x, down y, front z

    ---------------------------
    camera -> image2 projection
    ---------------------------

        y_image2 = P2_rect * x_rect
        y_image2 = P2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P2_rect = [f2_u,  0,      c2_u,  -f2_u b2_x;
                    0,    f2_v,   c2_v,  -f2_v b2_y;
                    0,    0,      1,     0]
                 = K * [1|t]

    """

    def __init__(self, calib_filepath):

        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])
        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    @staticmethod
    def read_calib_file(filepath):
        """
        Read in a calibration file and parse into a dictionary.
        :param filepath:
        :return:
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    ###############
    # projections #
    ###############

    @staticmethod
    def cart2hom(pts_3d):
        """
        Transform cartesian coordinates to homogeneous coordinates.
        :param pts_3d: nx3 points in Cartesian
        :return: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ---------
    # 3d to 3d
    # ---------

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        return np.transpose(np.dot(np.linalg.inv(self.R0),
                                   np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # --------
    # 3d to 2d
    # --------

    def project_ref_to_image(self, pts_3d_ref):
        pts_3d_rect = self.project_ref_to_rect(pts_3d_ref)
        return self.project_rect_to_image(pts_3d_rect)

    def project_rect_to_image(self, pts_3d_rect):
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)


###########################
# Compute 3D Bounding Box #
###########################


def compute_box_3d(label, P, scale=1.0):
    """
    Takes an object and a projection matrix (P) and projects the 3d bounding
    box into the image plane.
    :param label: 3d label object
    :param P: projection matrix
    :param scale: scale the bounding box
    :return: corners_2d: (8,2) array in left image coord.
             corners_3d: (8,3) array in in rect camera coord.
    """

    # compute rotational matrix around yaw axis
    R = rot_y(label.ry)

    # 3d bounding box dimensions in camera coordinates
    length = label.length * scale
    width = label.width * scale
    height = label.height * scale
    # 3d bounding box corners
    x_corners = [length / 2, length / 2, -length / 2, -length / 2, length / 2,
                 length / 2, -length / 2, -length / 2]
    y_corners = [-height, -height, -height, -height, 0, 0, 0, 0]
    z_corners = [-width / 2, width / 2, width / 2, -width / 2, -width / 2,
                 width / 2, width / 2, -width / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + label.t[0]
    corners_3d[1, :] = corners_3d[1, :] + label.t[1]
    corners_3d[2, :] = corners_3d[2, :] + label.t[2]

    # only draw 3d bounding box for objects in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    return corners_2d, np.transpose(corners_3d)


#####################
# Drawing Functions #
#####################

# draw projected 3D bounding box
def draw_projected_box_3d(image, corners_2d, color=(0, 255, 0), thickness=2):
    """
    Draw 3d bounding box in image
            2 -------- 1
           /|         /|
          3 -------- 0 .
          | |        | |
          . 6 -------- 5
          |/         |/
          7 -------- 4
    :param image: image on which the bounding box will be drawn
    :param corners_2d: (8,2) array of vertices for the 3d box in following order
    :param color
    :param thickness
    :return: image with bounding box
    """

    corners_2d = corners_2d.astype(np.int32)

    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        image = cv2.line(image, (corners_2d[i, 0], corners_2d[i, 1]),
                         (corners_2d[j, 0], corners_2d[j, 1]), color,
                         thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        image = cv2.line(image, (corners_2d[i, 0], corners_2d[i, 1]),
                         (corners_2d[j, 0], corners_2d[j, 1]), color,
                         thickness)
        i, j = k, k + 4
        image = cv2.line(image, (corners_2d[i, 0], corners_2d[i, 1]),
                         (corners_2d[j, 0], corners_2d[j, 1]), color,
                         thickness)

    return image


# draw projected BEV bounding box
def draw_projected_box_bev(image, corners_3d, color=(0, 255, 0), thickness=1,
                           confidence_score=None):
    """
     Draw BEV bounding box on image
     :param image: bev image of observable area
     :param corners_3d: corners of 3D bounding box in camera coordinates
     :param color:
     :param thickness:
     :param confidence_score: optional confidence score of the prediction to
     be displayed next to the bounding box
     :return: image with BEV bounding box
     """

    # extract corners of BEV bounding box either already transformed to
    # x/y-coordinates or from camera coordinates
    if corners_3d.shape[1] == 3:
        corners_x = corners_3d[:4, 0]
        corners_y = corners_3d[:4, 2]
    else:
        corners_x = corners_3d[:4, 0]
        corners_y = corners_3d[:4, 1]

    # convert coordinates from m to image coordinates
    pixel_corners_x = ((corners_x - VOX_Y_MIN) //
                       VOX_Y_DIVISION).astype(np.int32)
    pixel_corners_y = (INPUT_DIM_0 - ((corners_y - VOX_X_MIN) //
                                      VOX_X_DIVISION)).astype(np.int32)

    # draw BEV bounding box and mark front in different color
    image = cv2.line(image, (pixel_corners_x[3], pixel_corners_y[3]),
                     (pixel_corners_x[0], pixel_corners_y[0]), color,
                     thickness)
    image = cv2.line(image, (pixel_corners_x[0], pixel_corners_y[0]),
                     (pixel_corners_x[1], pixel_corners_y[1]), color,
                     thickness + 1)
    image = cv2.line(image, (pixel_corners_x[1], pixel_corners_y[1]),
                     (pixel_corners_x[2], pixel_corners_y[2]), color,
                     thickness)
    image = cv2.line(image, (pixel_corners_x[2], pixel_corners_y[2]),
                     (pixel_corners_x[3], pixel_corners_y[3]), color,
                     thickness)

    # display confidence score if provided
    if confidence_score:
        text = str(round(confidence_score, 2))  # round confidence score to
        # two decimal digits
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        size = cv2.getTextSize(text, font, font_scale, thickness)
        # get the size of the displayed text
        pos_x = int(np.mean(pixel_corners_x)) - (size[0][0] // 2)
        # center text in x-center of bounding box
        pos_y = np.min(pixel_corners_y) - 5
        # display text slightly above the bounding box
        cv2.putText(image, text, (pos_x, pos_y), font, font_scale, color=color)

    return image


# draw BEV image
def draw_bev_image(point_cloud):
    """
    Generate a grayscale image displaying a BEV perspective of the point cloud
    :param point_cloud: voxelized point cloud | shape: [length, width, height]
    :return: numpy array | shape: [length, width, 3]
    """

    bev_image = np.sum(point_cloud, axis=2)
    bev_image = bev_image - np.min(bev_image)
    divisor = np.max(bev_image) - np.min(bev_image)
    bev_image = 255 - (bev_image / divisor * 255)
    bev_image = np.dstack((bev_image, bev_image, bev_image)).astype(np.uint8)

    # add gray area to indicate camera FOV
    bev_image_bg = copy.copy(bev_image)
    triangle_cnt1 = np.array([(400, 700), (0, 700), (0, 300)])
    bev_image_bg = cv2.drawContours(bev_image_bg, [triangle_cnt1], 0,
                                    (120, 120, 120), -1)
    triangle_cnt2 = np.array([(400, 700), (800, 700), (800, 300)])
    bev_image_bg = cv2.drawContours(bev_image_bg, [triangle_cnt2], 0,
                                    (120, 120, 120), -1)

    # overlay background image to indicate FOV
    alpha = 0.2
    bev_image = cv2.addWeighted(bev_image_bg, alpha, bev_image, 1 - alpha, 0,
                                bev_image)

    return bev_image


####################
# Reader Functions #
####################

# read label file
def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    lines = [line for line in lines if line.split(' ')[0] != 'DontCare']
    objects = [Object3D(line) for line in lines]
    return objects


# read lidar scan
def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return PointCloud(xyz=scan[:, :-1], feats=scan[:, -1:])


# read image
def get_image(img_filename):
    return cv2.imread(img_filename)


####################
# Helper Functions #
####################

# invert transformation matrix
def inverse_rigid_trans(tr):
    """
    Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    :param tr: transformation matrix
    :return: inverse transformation matrix
    """
    inv_tr = np.zeros_like(tr)  # 3x4
    inv_tr[0:3, 0:3] = np.transpose(tr[0:3, 0:3])
    inv_tr[0:3, 3] = np.dot(-np.transpose(tr[0:3, 0:3]), tr[0:3, 3])
    return inv_tr


# rotation around y-axis
def rot_y(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


# project 3D points to image plane
def project_to_image(pts_3d, P):
    """
    Project 3d points to image plane.
    :param pts_3d: nx3 matrix
    :param P: 3x4 projection matrix
    :return: nx2 matrix
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


###########################
# Down-sample Point Cloud #
###########################

def downsample_by_average_voxel(point_cloud, voxel_size):
    """
    Voxel down-sampling using average function.
    points: a PointCloud named tuple containing "xyz" and "feats"
    voxel_size: float | side length of voxel cells used for down-sampling
    """

    # Create voxel grid
    xmin, ymin, zmin = np.amin(point_cloud.xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_idx = ((point_cloud.xyz - xyz_offset) // voxel_size).astype(np.int32)
    dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1

    # Get ordered array of voxel keys
    keys = xyz_idx[:, 0] + xyz_idx[:, 1] * dim_x + xyz_idx[:,
                                                   2] * dim_y * dim_x
    order = np.argsort(keys)
    keys = keys[order]

    # Order points
    xyz_ordered = point_cloud.xyz[order]

    # Get number of unique occupied voxels and number of points per voxel
    unique_keys, points_per_voxel = np.unique(keys, return_counts=True)

    # Get list of indices that correspond to the first point in the respective
    # voxel
    new_voxel_indices = np.hstack([[0], points_per_voxel[:-1]]).cumsum()

    # Down-sample point cloud by computing the mean over all points inside
    # a non-empty voxel
    point_cloud.xyz = np.add.reduceat(
        xyz_ordered, new_voxel_indices, axis=0) / points_per_voxel[:,
                                                  np.newaxis]

    if point_cloud.feats is not None:
        feats_ordered = point_cloud.feats[order]
        point_cloud.feats = np.add.reduceat(
            feats_ordered, new_voxel_indices, axis=0) / points_per_voxel[:,
                                                        np.newaxis]

    return point_cloud


# Filter out points outside the camera FoV
def filter_point_cloud(point_cloud, image, calib):
    """
    Get camera points that are visible in image and append image color
    to the points as attributes.
    """

    # Get all points with minimum distance of 10cm to camera
    fov_inds = point_cloud.xyz[:, 2] > 0.1
    fov_xyz, fov_feats = point_cloud[fov_inds, :]

    # Get image dimensions
    height, width = image.shape[0:2]

    # Project FoV points to image plane
    img_points = calib.project_ref_to_image(fov_xyz)

    # Get indices of all points with projections to the image plane
    # inside the image boundaries
    img_inds = np.logical_and.reduce(
        [img_points[:, 0] > 0, img_points[:, 0] < width,
         img_points[:, 1] > 0, img_points[:, 1] < height])

    # Assign image points to point cloud
    point_cloud.xyz = fov_xyz[img_inds, :]
    point_cloud.feats = fov_feats[img_inds, :]

    return point_cloud


# Get RGB features
def get_rgb_feats(point_cloud, image):
    # Get RGB color of pixels corresponding to remaining points
    rgb = image[np.int32(point_cloud.xyz[:, 1]),
          np.int32(point_cloud.xyz[:, 0]), ::-1].astype(np.float32) / 255

    # Add RGB information as point cloud features
    point_cloud.feats = np.concatenate((point_cloud.feats, rgb), axis=1)

    return point_cloud


# Construct graph
def construct_graph(points_xyz, radius, num_neighbors,
                    neighbors_downsample_method='random'):
    """Generate a local graph by radius neighbors.
    """
    # Compute nearest neighbors and get list of neighbor indices for each
    # point
    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree',
                            n_jobs=1).fit(points_xyz)
    indices = nbrs.radius_neighbors(points_xyz, return_distance=False)

    # Randomly sample maximum number of neighbors
    indices = [neighbors if neighbors.size <= num_neighbors else
               np.random.choice(neighbors, num_neighbors, replace=False)
               for neighbors in indices]

    # Construct edges
    vertices_v = np.concatenate(indices)
    vertices_i = np.concatenate([i * np.ones(neighbors.size, dtype=np.int32)
                                 for i, neighbors in enumerate(indices)])
    edges = np.array([vertices_v, vertices_i])

    return edges


# Assign labels to points
def get_point_labels_car(labels, xyz, expend_factor):
    """
    Assign class label and bounding boxes to xyz points.
    """
    num_points = xyz.shape[0]
    assert num_points > 0, "No point No prediction"
    assert xyz.shape[1] == 3
    # define label map
    label_map = {
        'Background': 0,
        'Car': 1,
        'DontCare': 3
    }
    # by default, all points are assigned with background label 0.
    cls_labels = np.zeros((num_points, 1), dtype=np.int64)
    # 3d boxes for each point
    boxes_3d = np.zeros((num_points, 1, 7))
    valid_boxes = np.zeros((num_points, 1, 1), dtype=np.float32)
    # add label for each object
    for label in labels:

        # Get object type
        object_type = label.type

        # Get class id
        class_id = label_map.get(object_type, 3)

        if object_type == 'Car':

            # Get all points inside object
            mask = get_object_points(label, xyz, expend_factor)
            yaw = label.ry
            while yaw < -0.25 * np.pi:
                yaw += np.pi
            while yaw > 0.75 * np.pi:
                yaw -= np.pi
            if yaw < 0.25 * np.pi:
                # horizontal
                cls_labels[mask, :] = class_id
                boxes_3d[mask, 0, :] = (label.t[0], label.t[1], label.t[2],
                                        label.length, label.height,
                                        label.width, label.ry)
                valid_boxes[mask, 0, :] = 1
            else:
                # vertical
                cls_labels[mask, :] = class_id + 1
                boxes_3d[mask, 0, :] = (label.t[0], label.t[1], label.t[2],
                                        label.length, label.height,
                                        label.width, label.ry)
                valid_boxes[mask, 0, :] = 1
        else:
            if object_type != 'DontCare':
                mask = get_object_points(label, xyz, expend_factor)
                cls_labels[mask, :] = class_id
                valid_boxes[mask, 0, :] = 0.0

    return cls_labels, boxes_3d, valid_boxes, label_map


def get_object_points(self, label, xyz, expend_factor=(1.0, 1.0, 1.0)):
    """Select points in a 3D box.
    Args:
        label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
        "height", "width", "lenth".
    Returns: a bool mask indicating points inside a 3D box.
    """

    normals, lower, upper = self.box3d_to_normals(label, expend_factor)
    projected = np.matmul(xyz, np.transpose(normals))
    points_in_x = np.logical_and(projected[:, 0] > lower[0],
                                 projected[:, 0] < upper[0])
    points_in_y = np.logical_and(projected[:, 1] > lower[1],
                                 projected[:, 1] < upper[1])
    points_in_z = np.logical_and(projected[:, 2] > lower[2],
                                 projected[:, 2] < upper[2])
    mask = np.logical_and.reduce((points_in_x, points_in_y, points_in_z))
    return mask


def box3d_to_normals(label, expend_factor=(1.0, 1.0, 1.0)):
    """Project a 3D box into camera coordinates, compute the center
    of the box and normals.
    Args:
        label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
        "height", "width", "lenth".
    Returns: a numpy array [3, 3] containing [wx, wy, wz]^T, a [3] lower
        bound and a [3] upper bound.
    """
    box3d_corners_xyz = box3d_to_cam_points(label, expend_factor)
    wx = box3d_corners_xyz[[0], :] - box3d_corners_xyz[[4], :]
    lx = np.matmul(wx, box3d_corners_xyz[4, :])
    ux = np.matmul(wx, box3d_corners_xyz[0, :])
    wy = box3d_corners_xyz[[0], :] - box3d_corners_xyz[[1], :]
    ly = np.matmul(wy, box3d_corners_xyz[1, :])
    uy = np.matmul(wy, box3d_corners_xyz[0, :])
    wz = box3d_corners_xyz[[0], :] - box3d_corners_xyz[[3], :]
    lz = np.matmul(wz, box3d_corners_xyz[3, :])
    uz = np.matmul(wz, box3d_corners_xyz[0, :])
    return (np.concatenate([wx, wy, wz], axis=0),
            np.concatenate([lx, ly, lz]), np.concatenate([ux, uy, uz]))


def box3d_to_cam_points(label, expend_factor=(1.0, 1.0, 1.0)):
    """Project 3D box into camera coordinates.
    Args:
        label: a dictionary containing "x3d", "y3d", "z3d", "yaw", "height"
            "width", "length".
    Returns: a numpy array [8, 3] representing the corners of the 3d box in
        camera coordinates.
    """

    yaw = label['yaw']
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    h = label['height']
    delta_h = h * (expend_factor[0] - 1)
    w = label['width'] * expend_factor[1]
    l = label['length'] * expend_factor[2]
    corners = np.array([[l / 2, delta_h / 2, w / 2],  # front up right
                        [l / 2, delta_h / 2, -w / 2],  # front up left
                        [-l / 2, delta_h / 2, -w / 2],  # back up left
                        [-l / 2, delta_h / 2, w / 2],  # back up right
                        [l / 2, -h - delta_h / 2, w / 2],
                        # front down right
                        [l / 2, -h - delta_h / 2, -w / 2],
                        # front down left
                        [-l / 2, -h - delta_h / 2, -w / 2],
                        # back down left
                        [-l / 2, -h - delta_h / 2,
                         w / 2]])  # back down right
    r_corners = corners.dot(np.transpose(R))
    tx = label['x3d']
    ty = label['y3d']
    tz = label['z3d']
    cam_points_xyz = r_corners + np.array([tx, ty, tz])
    return cam_points_xyz


# Box encoding
def voxelnet_box_encoding(cls_labels, points_xyz, boxes_3d):
    # offset
    boxes_3d[:, 0] = boxes_3d[:, 0] - points_xyz[:, 0]
    boxes_3d[:, 1] = boxes_3d[:, 1] - points_xyz[:, 1]
    boxes_3d[:, 2] = boxes_3d[:, 2] - points_xyz[:, 2]
    # Car
    mask = cls_labels[:, 0] == 2
    boxes_3d[mask, 0] = boxes_3d[mask, 0]/3.9
    boxes_3d[mask, 1] = boxes_3d[mask, 1]/1.56
    boxes_3d[mask, 2] = boxes_3d[mask, 2]/1.6
    boxes_3d[mask, 3] = np.log(boxes_3d[mask, 3]/3.9)
    boxes_3d[mask, 4] = np.log(boxes_3d[mask, 4]/1.56)
    boxes_3d[mask, 5] = np.log(boxes_3d[mask, 5]/1.6)
    # Pedestrian and Cyclist
    mask = (cls_labels[:, 0] == 1) + (cls_labels[:, 0] == 3)
    boxes_3d[mask, 0] = boxes_3d[mask, 0]/0.8
    boxes_3d[mask, 1] = boxes_3d[mask, 1]/1.73
    boxes_3d[mask, 2] = boxes_3d[mask, 2]/0.6
    boxes_3d[mask, 3] = np.log(boxes_3d[mask, 3]/0.8)
    boxes_3d[mask, 4] = np.log(boxes_3d[mask, 4]/1.73)
    boxes_3d[mask, 5] = np.log(boxes_3d[mask, 5]/0.6)
    # normalize all yaws
    boxes_3d[:, 6] = boxes_3d[:, 6]/(np.pi*0.5)
    return boxes_3d

def voxelnet_box_decoding(cls_labels, points_xyz, encoded_boxes):
    # Car
    mask = cls_labels[:, 0] == 2
    encoded_boxes[mask, 0] = encoded_boxes[mask, 0]*3.9
    encoded_boxes[mask, 1] = encoded_boxes[mask, 1]*1.56
    encoded_boxes[mask, 2] = encoded_boxes[mask, 2]*1.6
    encoded_boxes[mask, 3] = np.exp(encoded_boxes[mask, 3])*3.9
    encoded_boxes[mask, 4] = np.exp(encoded_boxes[mask, 4])*1.56
    encoded_boxes[mask, 5] = np.exp(encoded_boxes[mask, 5])*1.6
    # Pedestrian and Cyclist
    mask = (cls_labels[:, 0] == 1) + (cls_labels[:, 0] == 3)
    encoded_boxes[mask, 0] = encoded_boxes[mask, 0]*0.8
    encoded_boxes[mask, 1] = encoded_boxes[mask, 1]*1.73
    encoded_boxes[mask, 2] = encoded_boxes[mask, 2]*0.6
    encoded_boxes[mask, 3] = np.exp(encoded_boxes[mask, 3])*0.8
    encoded_boxes[mask, 4] = np.exp(encoded_boxes[mask, 4])*1.73
    encoded_boxes[mask, 5] = np.exp(encoded_boxes[mask, 5])*0.6
    # offset
    encoded_boxes[:, 0] = encoded_boxes[:, 0] + points_xyz[:, 0]
    encoded_boxes[:, 1] = encoded_boxes[:, 1] + points_xyz[:, 1]
    encoded_boxes[:, 2] = encoded_boxes[:, 2] + points_xyz[:, 2]
    # recover all yaws
    encoded_boxes[:, 6] = encoded_boxes[:, 6]*(np.pi*0.5)
    return encoded_boxes
