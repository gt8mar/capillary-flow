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
import cv2, json
import random
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
# from detectron2.data import MetadataCatalog
from fvcore.transforms.transform import PadTransform

# get data in the correct format
# train shit lol

config_file_path = "C:\\Users\\gt8mar\\detectron2\\detectron2\\model_zoo\\configs\\COCO-InstanceSegmentation\\mask_rcnn_R_50_FPN_3x.yaml"
json_train = "C:\\Users\\gt8mar\\capillary-flow\\tests\\230323_train.json"
json_val = "C:\\Users\\gt8mar\\capillary-flow\\tests\\230323_val.json"
folder_train = "D:\\Marcus\\train_backgrounds_export"
folder_val = "D:\\Marcus\\val_export"
# register_coco_instances("set_02", {}, path_to_masks, path_to_images)
register_coco_instances("my_dataset_train", {}, json_train, folder_train)
register_coco_instances("my_dataset_val", {}, json_val, folder_val)

dataset_train = load_coco_json(json_train, folder_train, "my_dataset_train")
dataset_val = load_coco_json(json_val, folder_val, "my_dataset_val")

# print(len(dataset))

# for d in ['train', 'val']:
#     register_coco_instances("set_02", {}, path_to_masks, path_to_images)


# for d in random.sample(dataset_train,3):
#     img = cv2.imread(d["file_name"])
#     print(img.shape)
#     visualizer = Visualizer(img[:,:,::-1], scale = 0.5)
#     out = visualizer.draw_dataset_dict(d)
#     plt.imshow(out.get_image()[:,:,::-1])
#     plt.show()


# test_path = "C:\\Users\\gt8mar\\Downloads"
# im = cv2.imread(test_path)
# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# print(outputs)



cfg = get_cfg()
# print(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(config_file_path)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# print(cfg.MODEL.WEIGHTS)
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

# model = detectron2.modeling.build_model(cfg)
# checkpointer = detectron2.checkpoint.DetectionCheckpointer(model)
# checkpointer.load(cfg.MODEL.WEIGHTS)


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()