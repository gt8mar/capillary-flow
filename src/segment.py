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

def main():
    json_train = "/hpc/mydata/marcus.forst/230323_train.json"
    json_val = "/hpc/mydata/marcus.forst/230323_val.json"
    folder_train = "/hpc/mydata/marcus.forst/train_backgrounds_export"
    folder_val = "/hpc/mydata/marcus.forst/val_export"
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # This is different than in train_detectron2.py:
    cfg.MODEL.WEIGHTS = os.path.join(weights_path, "model_final.pth")  #os.path.join(weights_path,"model_final.pth")  # original input: cfg.OUTPUT_DIR
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
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imsave(os.path.join("/hpc/mydata/marcus.forst/segmented", sample + "_segmented.png"), 
                    out.get_image()[:, :, ::-1])

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