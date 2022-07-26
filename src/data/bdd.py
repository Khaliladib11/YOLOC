# Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import json


import torch
from torch.utils import data
import torchvision.transforms as transforms


class BDD(data.Dataset):

    """
    Dataset class for BDD100K dataset to be used in this project.
    It offers a lot of flexibility
    """

    def __init__(self,
                 image_root: Path,
                 stage: str,
                 detection_classes: list,
                 det_task: bool,
                 de_seg_task: bool,
                 ll_seg_task: bool):
        """
        Constarcture
        :param image_root: Path for the data
        :param stage: to specify in which stage
        :param detection_classes: A list of classes to detect
        :param det_task: Bool value to enable or disable detection task
        :param de_seg_task: Bool value to enable or disable drivable area segmentation task
        :param ll_seg_task: Bool value to enable or disable Lane segmentation task
        """

        assert stage in ['train', 'test'], "Please stage must be: 'train' or  'test'."
        assert type(det_task) is bool, "de_task is bool parameter."
        assert type(de_seg_task) is bool, "de_seg_task is bool parameter."
        assert type(ll_seg_task) is bool , "ll_seg_task is bool parameter."

        self.root = image_root
        self.stage = stage
        self.detection_classes = detection_classes
        self.det_task = det_task
        self.de_seg_task = de_seg_task
        self.ll_seg_task = ll_seg_task

        self.image_path = self.root / Path('images/bdd100k/images/100k') / self.stage

        if self.det_task:
            self.det_path = self.root / Path('det_annotations') / self.stage

        if self.de_seg_task:
            self.de_seg_path = self.root / Path('da_seg_annotations') / self.stage

        if self.ll_seg_task:
            self.ll_seg_path = self.root / Path('ll_seg_annotations') / self.stage


        self.images = list(self.image_path.glob('**/*.jpg'))


        self.cls2idx = {}
        self.idx2cls = {}

        for idx in range(len(self.detection_classes)):
            self.cls2idx[self.detection_classes[idx]] = idx
            self.idx2cls[idx] = self.detection_classes[idx]


    def __get_image_name(self, idx):
        assert -1 < idx < len(self.images), "index out of range."
        return str(self.images[idx]).split('\\')[-1].split('.')[0]  # to get the name of the file

    # method to return the image
    def get_image(self, idx):
        assert -1 < idx < len(self.images), "index out of range."
        image_name = self.__get_image_name(idx)
        image = Image.open(self.image_path / Path(image_name + '.jpg'))
        return image

    # method to get the detection annotation
    def get_detection_annotation(self, idx):
        assert -1 < idx < len(self.images), "index out of range."
        image_name = self.__get_image_name(idx)

        label = {}
        classes = []
        bboxes = []

        file = open(self.det_path / Path(image_name + '.json'))

        annotation = json.load(file)

        objects = annotation['frames'][0]['objects']
        for obj in objects:
            if obj['category'] in self.detection_classes:
                x1 = obj['box2d']['x1']
                y1 = obj['box2d']['y1']
                x2 = obj['box2d']['x2']
                y2 = obj['box2d']['y2']

                bbox = [x1, y1, x2-x1, y2-y1]  # bbox of form: (x, y, w, h)

                if obj['category'] == 'traffic light':
                    cls = "traffic light" + " " + obj['attributes']['trafficLightColor']
                else:
                    cls = obj['category']

                classes.append(cls)
                bboxes.append(bbox)


        file.close()

        label['labels'] = classes
        label['boxes'] = bboxes

        return label

    # method to get the drivable area mask
    def get_drivable_area_annotation(self, idx):
        assert -1 < idx < len(self.images), "index out of range."
        image_name = self.__get_image_name(idx)

        mask = Image.open(self.de_seg_path / Path(image_name+'.png')).convert('L')

        return mask

    # method to get the lane mask
    def get_lane_annotation(self, idx):
        assert -1 < idx < len(self.images), "index out of range."
        image_name = self.__get_image_name(idx)

        mask = Image.open(self.ll_seg_path / Path(image_name+'.png')).convert('L')

        return mask

    def display_image(self, idx, **kwargs):
        assert -1 < idx < len(self.images), "index out of range."
        image = self.get_image(idx)
        if boxes:
            annotations = self.get_detection_annotation(idx)
            fig, ax = plt.subplots()
            ax.imshow(image)
            for i in range(len(annotations['boxes'])):
                box = annotations['boxes'][i]
                rect = patches.Rectangle((box[0], box[1]), box[2], box[3], facecolor="none", edgecolor='r')
                ax.add_patch(rect)

        if de_seg:
            de_mask = self.get_drivable_area_annotation(idx)
            ax.imshow(de_mask, cmap='jet', alpha=0.5*(np.array(de_mask)>0)  )
            print(de_mask)

        if ll_seg:
            ll_mask = self.get_lane_annotation(idx)
            ax.imshow(ll_mask, cmap='jet', alpha=0.5 * (np.array(ll_mask) > 0))

        else:
            plt.imshow(image)

        plt.axis('off')
        plt.show()



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pass


class bddDataset(data.Dataset):

    def __init__(self, cfg, stage, relative_path='../', image_size=(640, 640), transform=None):

        assert stage in ['train', 'test'], "Please stage must be: 'train' or  'test'."
        self.cfg = cfg
        self.transform = transform
        img_root = Path(relative_path + self.cfg.DATASET.IMAGESROOT)
        label_root = Path(relative_path + self.cfg.DATASET.LABELROOT)
        mask_root = Path(relative_path + self.cfg.DATASET.MASKROOT)
        lane_root  = Path(relative_path + self.cfg.DATASET.LANEROOT)

        if stage == 'train':
            self.stage = self.cfg.DATASET.TRAIN
        elif stage == 'test':
            self.stage = self.cfg.DATASET.TEST


        self.img_root = img_root / self.stage
        self.label_root = label_root / self.stage
        self.mask_root = mask_root / self.stage
        self.lane_root = lane_root / self.stage

        # self.dataformat = self.cfg.DATASET.IMAGEFORMAT

        self.images = list(self.img_root.glob(f'**/*.{self.cfg.DATASET.IMAGEFORMAT}'))

        self.mask_list = self.mask_root.iterdir()


    def _get_db(self):
        pass


    def __len__(self):
        pass

    def __getitem__(self, item):
        pass