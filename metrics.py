import numpy

get_iou(bbox1, bbox2):
    # Coordinates of the area of intersection.
    ix1 = np.maximum(bbox1[0], bbox2[0])
    iy1 = np.maximum(bbox1[1], bbox2[1])
    ix2 = np.minimum(bbox1[2], bbox2[2])
    iy2 = np.minimum(bbox1[3], bbox2[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = bbox2[3] - bbox2[1] + 1
    gt_width = bbox2[2] - bbox2[0] + 1

    # Prediction dimensions.
    pd_height = bbox1[3] - bbox1[1] + 1
    pd_width = bbox1[2] - bbox1[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou
