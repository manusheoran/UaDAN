# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "DeepLesion_train": {
            "data_dir": "DeepLesion/Images_png",
            "split": "train",
            "ann_file": "DeepLesion/DL_info.csv",
        },
        "combined_train": {
            "data_dir": "combined/npy_images",
            "split": "train",
            "ann_file": "combined/combined_info.csv",
        },
        "lits_train": {
            "data_dir": "LiTS/npy_images",
            "split": "train",
            "ann_file": "LiTS/lits_info.csv",
        },
        "target_train_fewshot": {
            "data_dir": "LiTS/npy_images",
            "split": "train",
            "ann_file": "LiTS/lits_info.csv",
        },
        "target_train_pseudo": {
            "data_dir": "LiTS/npy_images",
            "split": "train",
            "ann_file": "LiTS/lits_info.csv",
        },
        "kits_train": {
            "data_dir": "KiTS/npy_images",
            "split": "train",
            "ann_file": "KiTS/kits_info.csv",
        },
        
        "ircadb_train": {
            "data_dir": "Ircadb/npy_images",
            "split": "train",
            "ann_file": "Ircadb/ircadb_info.csv",
        },
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "voc_2007_train_6cls_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/labels/voc2007_train_6cls.json"
        },
        "voc_2012_train_6cls_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/labels/voc2012_train_6cls.json"
        },
        "water_train_cocostyle": {
            "img_dir": "watercolor/JPEGImages",
            "ann_file": "watercolor/water_train.json"
        },
        "water_test_cocostyle": {
            "img_dir": "watercolor/JPEGImages",
            "ann_file": "watercolor/water_test.json"
        },
        "clipart_train_voc": {
            "data_dir": "clipart",
            "split": "train"
        },
        "clipart_test_voc": {
            "data_dir": "clipart",
            "split": "test"
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        "foggy_cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "foggy_cityscapes/images",
            "ann_file": "foggy_cityscapes/annotations/foggy_instancesonly_filtered_gtFine_train.json"
        },
        "foggy_cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "foggy_cityscapes/images",
            "ann_file": "foggy_cityscapes/annotations/foggy_instancesonly_filtered_gtFine_val.json"
        },
        "vistas_instanceonly_seg_train_cocostyle": {
            "img_dir": "vistas/images",
            "ann_file": "vistas/annotations/instancesonly_train.json"
        },
        "vistas_instanceonly_seg_val_cocostyle": {
            "img_dir": "vistas/images",
            "ann_file": "vistas/annotations/instancesonly_val.json"
        },

        'sim10k_cocostyle': {
                "img_dir": 'sim10k/JPEGImages',
                "ann_file": 'sim10k/car_instances.json'
        },
        'kitti_cocostyle': {
                "img_dir": 'kitti/training/image_2',
                "ann_file": 'kitti/annotations/caronly_training.json'
        },
        'cityscapes_car_train_cocostyle': {
                "img_dir": 'cityscapes/images',
                "ann_file": 'cityscapes/annotations/caronly_filtered_gtFine_train.json',
        },
        'cityscapes_car_val_cocostyle': {
                "img_dir": 'cityscapes/images',
                "ann_file": 'cityscapes/annotations/caronly_filtered_gtFine_val.json',
        },
        "voc_2007_train_watercolor_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/annotations/pascal_train2007.json"
        },
        "voc_2007_val_watercolor_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/annotations/pascal_val2007.json"
        },
        "voc_2012_train_watercolor_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/annotations/pascal_train2012.json"
        },
        "voc_2012_val_watercolor_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/annotations/Annotations/pascal_val2012.json"
        },
        'clipart_cocostyle': {
                "img_dir": 'clipart/JPEGImages',
                "ann_file": 'clipart/instances.json',
        },
        'watercolor_train_cocostyle': {
                "img_dir": 'watercolor/JPEGImages',
                "ann_file": 'watercolor/instances_train.json',
        },
        'watercolor_val_cocostyle': {
                "img_dir": 'watercolor/JPEGImages',
                "ann_file": 'watercolor/instances_test.json',
        },
        'synthia_cocostyle': {
            "img_dir": 'synthia_760/images',
            "ann_file": 'synthia_760/annotations/6cls_filtered_train.json'
        },
        'synthia100_crop640_cocostyle': {
            "img_dir": 'synthia_crop640_100/images',
            "ann_file": 'synthia_crop640_100/annotations_7cls_filtered/instancesonly_train.json'
        },
        'synthia_crop640_cocostyle': {
            "img_dir": 'synthia_crop640/images',
            "ann_file": 'synthia_crop640/annotations_7cls_filtered/instancesonly_train.json'
        },
        'cityscapes_6cls_train_cocostyle': {
            "img_dir": 'cityscapes/images',
            "ann_file": 'cityscapes/annotations/6cls_filtered_gtFine_train.json',
        },
        'cityscapes_6cls_val_cocostyle': {
            "img_dir": 'cityscapes/images',
            "ann_file": 'cityscapes/annotations/6cls_filtered_gtFine_val.json',
        },
        'cityscapes_6cls_val_1_cocostyle': {
            "img_dir": 'cityscapes/images',
            "ann_file": 'cityscapes/annotations/6cls_filtered_gtFine_val_1.json',
        },
    }

    @staticmethod
    def get(name):
        if "DeepLesion" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="DL_target_combinedDataset",
                args=args,
            )
        if "combined" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="combinedDataset_uda",
                args=args,
            )
        if "kits" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="KiTSDataset",
                args=args,
            )
        
        if "lits" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="LiTSDataset",
                args=args,
            )
        
        if "ircadb" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="ircadbDataset",
                args=args,
            )
        
        
        if "fewshot" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="combinedDataset_fewshot",
                args=args,
            )
        
        if "pseudo" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="combinedDataset_pseudo",
                args=args,
            )
        
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
