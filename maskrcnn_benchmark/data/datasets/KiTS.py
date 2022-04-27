import torch
import torchvision
import numpy as np
import os
import csv
import logging
import json

from maskrcnn_benchmark.data.datasets.load_ct_img_kits import load_prep_img
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.DeepLesion_utils import load_tag_dict_from_xlsfile, gen_mask_polygon_from_recist, load_lesion_tags
from maskrcnn_benchmark.data.datasets.DeepLesion_utils import gen_parent_list, gen_exclusive_list, gen_children_list
from maskrcnn_benchmark.data.transforms.build import build_d2transforms

class KiTSDataset(object):

    def __init__(
        self, split, data_dir, ann_file, transforms=None, is_source= True
    ):
        self.transforms = transforms
        self.split = split
        self.data_path = data_dir
        self.is_source = is_source
        self.classes = ['__background__',  # always index 0
                        'lesion']
        self.num_classes = len(self.classes)
        ann_file = "/data/nihcc/FCOS/csv_files_datasets/kits_test.csv"  #kits_train, kits_keyslice_test
        #ann_file = "/gdrive/MyDrive/TCS/KiTS/kits_test1.csv"
        self.loadinfo(ann_file)
        self.image_fn_list, self.lesion_idx_grouped = self.load_split_index()
        self.num_images = len(self.image_fn_list)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info('KiTS %s num_images: %d' % (split, self.num_images))

    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, info).
        """
        #image_fn = 'Copy of ' + self.image_fn_list[index]
        image_fn =  self.image_fn_list[index]
        lesion_idx_grouped = self.lesion_idx_grouped[index]

        boxes0 = self.boxes[lesion_idx_grouped]
        #print('lesion_idx_grouped',lesion_idx_grouped)
        
        # slice_no = self.slice_idx[lesion_idx_grouped][0]
        slice_intv = self.slice_intv[lesion_idx_grouped][0]
        spacing = self.spacing[lesion_idx_grouped][0]
        

        num_slice = cfg.INPUT.NUM_SLICES * cfg.INPUT.NUM_IMAGES_3DCE
        is_train = self.split=='train'
        if is_train and cfg.INPUT.DATA_AUG_3D is False:
            slice_radius = diameters.min() / 2 * spacing / slice_intv * abs(cfg.INPUT.DATA_AUG_3D)  # lesion should not be too small
            slice_radius = int(slice_radius)
            if slice_radius > 0:
                if cfg.INPUT.DATA_AUG_3D > 0:
                    delta = np.random.randint(0, slice_radius+1)
                else:  # give central slice higher prob
                    ar = np.arange(slice_radius+1)
                    p = slice_radius-ar.astype(float)
                    delta = np.random.choice(ar, p=p/p.sum())
                if np.random.rand(1) > .5:
                    delta = -delta

                dirname, slicename = image_fn.split(os.sep)
                slice_idx = int(slicename[:-4])
                image_fn1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
                if os.path.exists(os.path.join(self.data_path, image_fn1)):
                    image_fn = image_fn1
        im, im_scale, crop = load_prep_img(self.data_path, image_fn, spacing, slice_intv,
                                           cfg.INPUT.IMG_DO_CLIP, num_slice=num_slice, is_train=is_train)


        current_im_shape = im.shape[0:2]
        boxes_new = boxes0.copy()
        if cfg.INPUT.IMG_DO_CLIP:
            offset = [crop[2], crop[0]]
            boxes_new -= offset*2
        boxes_new *= im_scale

        # for i,box in enumerate(boxes_new):
        #   x1,y1,x2,y2 = box
        #   boxes_new[i,:] = x1,y1,x2+x1,y2+y1

        #print('boxes_new',boxes_new)
        im = im.astype(np.uint8)
        im,boxes_new =  build_d2transforms(im,boxes_new)
        im = im.copy() - cfg.INPUT.PIXEL_MEAN
        im = torch.from_numpy(im.transpose((2, 0, 1))).to(dtype=torch.float)



        boxes = torch.as_tensor(boxes_new).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (im.shape[2], im.shape[1]), mode="xyxy")

        num_boxes = boxes.shape[0]
        classes = torch.ones(num_boxes, dtype=torch.int)  # lesion/nonlesion
        target.add_field("labels", classes)
        
        domain_labels = torch.ones_like(classes, dtype=torch.uint8) if self.is_source else torch.zeros_like(classes, dtype=torch.uint8)
        target.add_field("is_source", domain_labels)

  

        infos = {'im_index': index, 'lesion_idxs': lesion_idx_grouped, 'image_fn': image_fn, 
                 'crop': crop,  'slice_intv': slice_intv, 'im_scale': im_scale, 'spacing' : spacing, 'current_im_shape' : current_im_shape}
        return im, target, infos

    def __len__(self):
        return len(self.image_fn_list)

    def load_split_index(self):
        """
        need to group lesion indices to image indices, since one image can have multiple lesions
        :return:
        """

        split_list = ['train', 'val', 'test', 'small']
        index = split_list.index(self.split)
        if self.split != 'small':
            lesion_idx_list = np.where((self.train_val_test == index + 1) )[0]
        else:
            lesion_idx_list = np.arange(30)
        fn_list = self.filenames[lesion_idx_list]
        fn_list_unique, inv_ind = np.unique(fn_list, return_inverse=True)
        lesion_idx_grouped = [lesion_idx_list[inv_ind==i] for i in range(len(fn_list_unique))]
        return fn_list_unique, lesion_idx_grouped

    def loadinfo(self, path):
        """load annotations and meta-info from DL_info.csv"""
        info = []
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                filename = row[0]  # replace the last _ in filename with / or \
                #print(filename)
                #idx = filename.rindex('_')
                row[0] = filename
                info.append(row)
        info = info[1:]

        # the information not used in this project are commented
        self.filenames = np.array([row[0] for row in info])
        
        self.slice_idx = np.array([int(row[1]) for row in info])
        
        self.boxes = np.array([[float(x) for x in row[2].split('[')[-1].split(']')[0].split(',')] for row in info])
        #self.boxes -= 1  # coordinates in info file start from 1
       
        self.spacing = np.array([row[4] for row in info])
        self.slice_intv = np.array([row[3] for row in info])
        
        self.cr_image_size = np.array([[int(x) for x in row[5].split('(')[-1].split(')')[0].split(',')] for row in info])
        
        self.train_val_test = np.array([int(row[6]) for row in info])
