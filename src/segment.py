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
import numpy as np
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

def folder_into_COCO(path):
        """
        Takes a folder of images and converts it into a COCO dataset.

        Args:
            path (str): Path to the folder containing the images to be segmented.
        
        Returns:
            json_path (str): Path to the json file containing the COCO dataset.
        """
        # Create a json file for the dataset
        date_experiment = time.strftime("%Y%m%d")
        json_path = os.path.join(path, f'dataset_{date_experiment}.json')
        with open(json_path, 'w') as f:
            f.write('{"images": [')
            for i, file in enumerate(os.listdir(path)):
                if file.endswith('.tiff'):
                    numpy_image = cv2.imread(os.path.join(path, file))
                    height = numpy_image.shape[0]
                    width = numpy_image.shape[1]
                    # print(numpy_image.shape)
                    f.write('{"file_name": "' + file + '", "height": ' + str(height) + ', "width": ' + str(width) + '1280, "id": ' + str(i) + '},')
            f.write('], "categories": [{"supercategory": "background", "id": 0, "name": "background"}], "annotations": []}')
        return json_path

def parse_filename(filename):
    """
    Parses the filename of an image into its participant, date, and video number.

    Args:
        filename (str): Filename of the image. format: set_participant_date_video_background.tiff
    
    Returns:
        participant (str): Participant number.
        date (str): Date of the video.
        video (str): Video number.
    """
    filename_no_ext = filename.split('.')[0]
    participant = filename_no_ext.split('_')[-4]
    date = filename_no_ext.split('_')[-3]
    video = filename_no_ext.split('_')[-2]
    return participant, date, video

def parse_COCO(json_path):
    """
    Parses the COCO dataset into a dictionary containing the participant, date, and video number for each image.

    Args:
        json_path (str): Path to the json file containing the COCO dataset.
    
    Returns:
        dict: Dictionary containing the participant, date, and video number for each image.    
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    dict = {}
    for image in data['images']:
        filename = image['file_name']
        participant, date, video = parse_filename(filename)
        dict[filename] = [participant, date, video]
    return dict

def COCO_filename_remove_contrast(json_path):
    """
    Checks to see if the filename has _contrast in it and changes the filename in the COCO dataset to match.  

    Args:
        json_path (str): Path to the json file containing the COCO dataset.
    Returns:
        None  
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    for image in data['images']:
        if image['file_name'].endswith('_contrast.tiff'):
            image['file_name'] = image['file_name'].replace('_contrast', '')   
    with open(json_path, 'w') as f:
        json.dump(data, f)

def main(path='/hpc/projects/capillary-flow/results/backgrounds', verbose = False, verbose_plot = False):
    """
    Uses detectron2 to segment images using instance segmentation inference.
    Saves the results to a folder called "segmentation_results" in the same
    folder as the images.

    Args:
        path (str): Path to the folder containing the images to be segmented.
        verbose (bool): Whether to plot the results or not.
    
    Returns:
        mask_dict (dict): Dictionary containing the masks for each image.
    
    Saves:
        mask_int (png): Segmented images in png format.
    """
    # Create a detectron2 config and a detectron2 default predictor    
    json_path = "/hpc/projects/capillary-flow/results/dataset_json/dataset_230626.json"
    COCO_filename_remove_contrast(json_path)
    folder_seg = path
    # json_seg = folder_into_COCO(folder_seg)
    weights_path = "/home/marcus.forst/output"
    register_coco_instances("my_dataset_seg", {}, json_path, folder_seg)
    dataset_seg = load_coco_json(json_path, folder_seg, "my_dataset_seg")

    # Begin inference
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask" # This was a necessary addition for my segmentation files to run
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("my_dataset_train",)
    # cfg.DATASETS.TEST = ()
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
    mask_dict = {}
    for d in dataset_seg:    
        # extract the filename from the path
        filename = os.path.basename(d["file_name"])
        # check if filename is in background folder
        if filename not in os.listdir(folder_seg):
            print(f"filename: {filename} not in folder: {folder_seg}")
        else:
            im = cv2.imread(d["file_name"]) 
            if verbose:       
                print(f"filename: {filename} has shape:")
                print(im.shape)
            participant, date, video = parse_filename(filename)
            # remove the file extension
            filename_without_ext = os.path.splitext(filename)[0]
            # extract the desired string from the filename
            sample = filename_without_ext.split('_background')[0]
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            if len(outputs["instances"].pred_masks) == 0:
                if verbose:
                    print("no masks found")
                mask = np.zeros((im.shape[0], im.shape[1]))
            else:
                total_mask = np.zeros((im.shape[0], im.shape[1]))
                for i in range(len(outputs["instances"].pred_masks)):
                    mask = outputs["instances"].pred_masks[i].cpu().numpy()
                    total_mask += mask
                if verbose:
                    print("The number of nonzero pixels in the mask is:")
                    print(np.count_nonzero(mask))
                if verbose_plot:
                    v = Visualizer(im[:, :, ::-1],
                                scale=0.5, 
                                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                    )
                    print(f'The number of classes is {outputs["instances"].pred_classes}')
                    print(f'The number of instances is {outputs["instances"].pred_boxes}')
                    print(im[:,:,-1])
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    plt.imsave(os.path.join(cfg.OUTPUT_DIR, str(d["file_name"]) + "_fancy_seg.png"), 
                                out.get_image()[:, :, ::-1])
                
            # Convert boolean array to integer array
            mask_int = total_mask.astype(int)
            mask_dict[sample] = mask_int
            # Save the mask
            os.makedirs(os.path.join("/hpc/projects/capillary-flow/data", participant, date, video, "D_segmented"), exist_ok=True)
            plt.imsave(os.path.join("/hpc/projects/capillary-flow/data", participant, date, video, "D_segmented", filename_without_ext + "_seg.png"), 
                        mask_int, cmap='gray')
            plt.imsave(os.path.join("/hpc/projects/capillary-flow/results/segmented", filename_without_ext + "_seg.png"), 
                        mask_int, cmap='gray')
                

            # # Save the integer array to a CSV file            
            # np.savetxt(os.path.join(cfg.OUTPUT_DIR, filename_without_ext + "_segs.csv"), mask_int, 
            #            delimiter=',', fmt='%s')
    return mask_dict

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    print("Running segment.py...")
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))