"""
Filename: segment_model.py
-------------------
This file trains a machine learning model to
perform instance segmentation on images. 
By: Marcus Forst
with help from: https://huggingface.co/blog/fine-tune-segformer
"""

import torch
import detectron2
import os
from detectron2.data.datasets import register_coco_instances

# get data in the correct format
# train shit lol

path_to_dataset = "C:\\Users\\gt8mar\\capillary-flow\\data\\processed\\set_02\\230322_export.json"
register_coco_instances("set_02", {}, path_to_dataset, "path/to/image/dir")