import sys

from operator import mod
from turtle import window_width
from detectron2.data.catalog import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2 import model_zoo
import torch
import myclasses
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
import numpy as np
from tempfile import TemporaryFile
import os

class Detector:
    def __init__(self, model_type= "OD"):

        self.cfg= get_cfg()
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS= model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
            #self.cfg.MODEL.WEIGHTS= os.path.join("/home/khaled.dawoud/Desktop/ATB/Detection_Part/detectron2/output_fasterrcnn_1/model_0004999.pth")
            # # self.cfg.merge_from_file(model_zoo.get_config_file("https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            # # self.cfg.MODEL.WEIGHTS= model_zoo.get_checkpoint_url("https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
            
        elif model_type == "IS":    
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS= model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
            # self.cfg.merge_from_file(model_zoo.get_config_file("https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
            # self.cfg.MODEL.WEIGHTS= model_zoo.get_checkpoint_url("https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
  

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE= "cuda"

        self.predictor= DefaultPredictor(self.cfg)

    def onImage(self, imagePath, type):
        if type=="IS":
            image = cv2.imread(imagePath)
            # image = cv2.resize(image, (256,256) , interpolation=cv2.INTER_AREA)

            predections = self.predictor(image)

            c = predections["instances"].pred_classes.tolist()
            index_final, class_num = myclasses.calculat_index(c)
            self.indesFinal = index_final
            new_instances = predections["instances"][index_final]
            new_mask = myclasses.claculate_mask(new_instances.pred_masks)



            from detectron2.utils.visualizer import ColorMode

            im = cv2.imread(imagePath)
            # im=  cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)

            outputs = self.predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                            instance_mode=ColorMode.IMAGE_BW
                            # remove the colors of unsegmented pixels. This option is only available for segmentation models
                            )
            out = v.draw_instance_predictions(outputs["instances"][index_final].to("cpu"))
            computed_image=out.get_image()[:, :, ::-1]


            return computed_image, new_mask, index_final , class_num , new_instances.scores

        elif  type=="OD":  
            image = cv2.imread(imagePath)
            predections = self.predictor(image)
            c = predections["instances"].pred_classes.tolist()

            from detectron2.utils.visualizer import ColorMode
            im = cv2.imread(imagePath)
            # im=  cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
            outputs = self.predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                            instance_mode=ColorMode.IMAGE
                            # remove the colors of unsegmented pixels. This option is only available for segmentation models
                            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            computed_image=out.get_image()

            return computed_image 

    def draw_image_with_boxes(self,filename, boxes_list):
        # load the image
        data = plt.imread(filename)
        # plot the image
        plt.imshow(data)
        # get the context for drawing boxes
        ax = plt.gca()
        # plot each box
        for box in boxes_list:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=5)
            # draw the box
            ax.add_patch(rect)
        # show the plot
        plt.show()


    def plot_det(self,path):

        from detectron2.utils.visualizer import ColorMode

        im = cv2.imread(path)
        # im=cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)

        outputs = self.predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                           instance_mode=ColorMode.SEGMENTATION
                           # remove the colors of unsegmented pixels. This option is only available for segmentation models
                           )
                           

        c = outputs["instances"].pred_classes.tolist()
        index_final, class_num = myclasses.calculat_index(c)
        self.indesFinal = index_final
        new_instances = outputs["instances"][index_final] 

        out = v.draw_instance_predictions(new_instances.to("cpu"))
        computed_image=out.get_image()[:, :, ::-1]
        return computed_image
        # plt.imshow(out.get_image()[:, :, :])
        # plt.show()



        


          



