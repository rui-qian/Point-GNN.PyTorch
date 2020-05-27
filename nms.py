import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


###############
# Compute IoU #
###############

def compute_bbox_iou(query_box, search_boxes):
    """
    Compute Intersection over Union (IoU) for 3D bounding boxes.

    :param query_box: 3D box corners of query box | shape: [8, 3]
    :param search_boxes: array of boxes to compute IoU with | shape: [N, 8, 3]
    """

    # Compute min and max values of boxes for each dimension
    query_box_max_corner = np.max(query_box, axis=0)
    query_box_min_corner = np.min(query_box, axis=0)
    search_boxes_max_corner = np.max(search_boxes, axis=1)
    search_boxes_min_corner = np.min(search_boxes, axis=1)

    # Create non-overlap mask
    non_overlap_mask = np.logical_or(
        query_box_max_corner < search_boxes_min_corner,
        query_box_min_corner > search_boxes_max_corner)
    non_overlap_mask = np.any(non_overlap_mask, axis=1)

    # Instantiate container for IoUs
    ious = np.zeros(search_boxes.shape[0])

    # Construct polygon in xz-plane for query box
    p1 = Polygon([(query_box[0, 0], query_box[0, 2]),
                  (query_box[1, 0], query_box[1, 2]),
                  (query_box[2, 0], query_box[2, 2]),
                  (query_box[3, 0], query_box[3, 2])])

    # Iterate over search boxes
    for i in range(search_boxes.shape[0]):

        # Check if boxes overlap
        if not non_overlap_mask[i]:

            # Get search box
            search_box = search_boxes[i]

            # Construct polygon for search box in xz-plane
            p2 = Polygon([(search_box[0, 0], search_box[0, 2]),
                          (search_box[1, 0], search_box[1, 2]),
                          (search_box[2, 0], search_box[2, 2]),
                          (search_box[3, 0], search_box[3, 2])])

            # Compute intersection
            inter_area = p1.intersection(p2).area

            # compute IoU of the two bounding boxes
            iou = inter_area / (p1.area + p2.area - inter_area)

            # Set IoU
            ious[i] = iou

    return ious


def boxes_3d_to_corners(boxes_3d):
    """
    Convert box parameters to 3D bounding box corner points.

    :param boxes_3d: numpy array containing box predictions | shape: (N, 7)

    :return all_corners: numpy array containing bounding box corners |
                         shape: (N, 8, 3)

    """

    # Container for all corners
    all_corners = []

    # Iterate over all bounding boxes
    for box_3d in boxes_3d:

        # Extract bounding box parameters
        x3d, y3d, z3d, l, h, w, yaw = box_3d

        # Construct rotation matrix
        R = np.array([[np.cos(yaw),  0,  np.sin(yaw)],
                      [0,            1,  0],
                      [-np.sin(yaw), 0,  np.cos(yaw)]])

        # Specify axis-aligned corner points
        corners = np.array([[l/2, 0.0, w/2],  # front up right
                            [l/2, 0.0, -w/2],  # front up left
                            [-l/2, 0.0, -w/2],  # back up left
                            [-l/2, 0.0, w/2],  # back up right
                            [l/2, -h,  w/2],  # front down right
                            [l/2, -h, -w/2],  # front down left
                            [-l/2, -h, -w/2],  # back down left
                            [-l/2, -h,  w/2]])  # back down right

        # Rotate corner points
        r_corners = corners.dot(np.transpose(R))

        # Translate corner points
        corners_3d = r_corners+np.array([x3d, y3d, z3d])

        # Append to list
        all_corners.append(corners_3d)

    # Stack corner arrays along first dimension
    all_corners = np.array(all_corners)

    return all_corners


def nms(cls_labels, cls_pred, box_predictions, nms_threshold=0.25):
    """Apply non-maximum suppression to bounding boxes.

    :param box_predictions: bounding box predictions | shape: (N, 7)
    :param cls_labels: class label | shape: (N, 1)
    :param cls_pred: class predictions | shape: (N, 1)
    :param nms_threshold: NMS threshold | float
    """

    # Sort predictions in descending order
    ordered_inds = np.argsort(cls_pred)[::-1]
    cls_pred = cls_pred[ordered_inds]
    cls_labels = cls_labels[ordered_inds]
    box_predictions = box_predictions[ordered_inds]

    # Convert box parameters to 3D corners
    box_corners = boxes_3d_to_corners(box_predictions)

    # Initialize box selector
    keep = np.ones((cls_pred.shape[0],), dtype=np.bool)

    # Iterate over all predictions
    for i in range(cls_pred.size-1):
        if keep[i]:

            # Only compute on the rest of boxes
            valid = keep[(i + 1):]

            # Computer overlap with other boxes
            overlap = compute_bbox_iou(box_corners[i], box_corners[(i+1):][valid])

            print(overlap.shape)
            # Get mask for all overlapping boxes of same class as query box
            remove_overlap = np.logical_and(
                overlap > nms_threshold,
                cls_labels[(i+1):][valid] == cls_labels[i])
            print(remove_overlap.shape)

            # Get all overlapping boxes of the same class
            box_group = np.concatenate(
                [box_predictions[(i+1):][valid][remove_overlap],
                 box_predictions[[i]]], axis=0)

            # Compute median box parameters
            median_box = np.median(box_group, axis=0)

            # Compute corners and IoU of median box
            median_box_corners = boxes_3d_to_corners(
                np.expand_dims(median_box, axis=0))
            median_box_iou = compute_bbox_iou(median_box_corners[0],
                                              box_corners[(i+1):]
                                              [valid][remove_overlap])

            # Update scores based on IoU with median box
            cls_pred[i] += np.sum(
                cls_pred[(i+1):][valid][remove_overlap]*median_box_iou)

            # Remove duplicate boxes
            keep[(i+1):][valid] = np.logical_not(remove_overlap)

    # Get indices of retained boxes
    inds = np.where(keep)

    # Get corresponding labels, scores and boxes of final detections
    final_cls_labels = cls_labels[inds]
    final_cls_pred = cls_pred[inds]
    final_box_predictions = box_predictions[inds]

    return final_cls_labels, final_cls_pred, final_box_predictions


if __name__ == '__main__':

    box_predictions = np.array([[0, 0, 0, 3, 1, 1, 0],
                                [0.5, 0, 0.5, 3, 1, 1, np.pi/4],
                                [0.5, 0, -0.5, 3, 1, 1, -np.pi/4],
                                [-0.5, 0, 0.5, 3, 1, 1, np.pi/4],
                                [-0.5, 0, -0.5, 3, 1, 1, -np.pi/4],
                                [0, 0, 0, 3, 1, 1, np.pi/2]])
    cls_preds = np.random.rand(6)
    cls_labels = np.ones((6,))

    final_cls_labels, final_cls_preds, final_boxes = nms(cls_labels, cls_preds,
                                                         box_predictions)
    print(final_cls_preds)
