"""
Filename: annotations_to_COCO.py
---------------------------------------
This file inputs annotations and images and outputs a COCO dataset in json format.

By: Marcus Forst
"""

import json
import os 
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from get_images import get_images
from load_image_array import load_image_array

# load masks
mask_dir = '/'
mask_name_list = get_images(mask_dir)
mask_list = load_image_array(mask_dir)
# make annotations of masks using find_contours

# make bounding boxes of contours

# load metadata into json file

# load annotations and bounding boxes into json file

labels_info = []
for mask in mask_list:
    # opencv 3.2
    mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
    # before opencv 3.2
    # contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
    #                                                    cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        # segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)
    if len(segmentation) == 0:
        continue
    # get area, bbox, category_id and so on
    labels_info.append(
        {
            "segmentation": segmentation,  # poly
            "area": area,  # segmentation area
            "iscrowd": 0,
            "image_id": index,
            "bbox": [x1, y1, bbox_w, bbox_h],
            "category_id": category_id,
            "id": label_id
        },
    )