# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
from maskrcnn_benchmark.data import transforms as T1
from maskrcnn_benchmark.data.transforms import detection_utils as utils
def build_d2transforms(image,bbox):
  transform_list = [
        T1.RandomFlip(prob=0.40, horizontal=False, vertical=True),
        T1.RandomFlip(prob=0.40, horizontal=True, vertical=False),
        T1.RandomExtent([0.8,1.2], [0.125,0.125]),
        T1.Resize((512,512))
    ]

  #print('inside trans',image.shape, bbox)
  image, transforms = T1.apply_transform_gens(transform_list, image)
 
  #dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

  annos = [
      utils.transform_instance_annotations(obj, transforms, image.shape[:2])
      for obj in bbox
  ]

  return image,annos
def build_transforms(cfg, is_train=True):
    if 'DeepLesion' in cfg.DATASETS.TRAIN[0]:
        return None 
    if 'lits' in cfg.DATASETS.TRAIN[0]:
        return None
    if 'DeepLesion' in cfg.DATASETS.SOURCE_TRAIN[0]:
        return None 
    if 'train' in cfg.DATASETS.TARGET_TRAIN[0]:
        return None 
    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            ))
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
