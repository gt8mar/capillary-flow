"""
Filename: segment.py
-------------------
This file uses a detectron2 trained machine learning model to
segment images using instance segmentation on a hpc (high 
performance computer). 
By: Marcus Forst
"""

import torch
import detectron2
import os, time
import cv2, json
import random
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import ColorMode
# from detectron2.data import MetadataCatalog

def main(path=None):
    json_train = "/hpc/mydata/marcus.forst/230323_train.json"
    json_val = "/hpc/mydata/marcus.forst/230323_val.json"
    folder_train = "/hpc/mydata/marcus.forst/train_backgrounds_export"
    folder_val = "/hpc/mydata/marcus.forst/val_export"

    # folder_seg = "/hpc/mydata/marcus.forst/capillary-flow/results/backgrounds"
    folder_seg = path
    
    # json_train = "D:\\Marcus\\segmentations\\230323_train.json"
    # json_val = "D:\\Marcus\\segmentations\\230323_val.json"
    # folder_train = "D:\\Marcus\\train_backgrounds_export"
    # folder_val = "D:\\Marcus\\val_export"
    # weights_path = "C:\\Users\\gt8mar\\capillary-flow\\output"
    
    weights_path = "/home/marcus.forst/output"
    register_coco_instances("my_dataset_train", {}, json_train, folder_train)
    register_coco_instances("my_dataset_val", {}, json_val, folder_val)

    dataset_train = load_coco_json(json_train, folder_train, "my_dataset_train")
    dataset_val = load_coco_json(json_val, folder_val, "my_dataset_val")

    # # This allows you to visualize your training segmentations
    # for d in random.sample(dataset_train,3):
    #     img = cv2.imread(d["file_name"])
    #     print(img.shape)
    #     visualizer = Visualizer(img[:,:,::-1], scale = 0.5)
    #     out = visualizer.draw_dataset_dict(d)
    #     plt.imshow(out.get_image()[:,:,::-1])
    #     plt.show()


    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask" # This was a necessary addition for my segmentation files to run
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # This is different than in train_detectron2.py:
    cfg.MODEL.WEIGHTS = os.path.join(weights_path, "model_final.pth")  #os.path.join(weights_path,"model_final.pth")  # original input: cfg.OUTPUT_DIR
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    predictor = DefaultPredictor(cfg)
    
    for d in dataset_val:    
        im = cv2.imread(d["file_name"])
        # extract the filename from the path
        filename = os.path.basename(d["file_name"])
        # remove the file extension
        filename_without_ext = os.path.splitext(filename)[0]
        # extract the desired string from the filename
        sample = filename_without_ext.split('_background')[0]
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        print(im[:,:,-1])
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imsave(os.path.join(cfg.OUTPUT_DIR, str(d["file_name"]) + "_segs.png"), 
                    out.get_image()[:, :, ::-1])
    return im[:,:,-1]

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))