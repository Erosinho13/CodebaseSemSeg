from .voc import VOCSegmentation
from .ade import AdeSegmentation
from .cityscapes import Cityscapes
from .transform import *
import random
from .utils import Subset
from functools import partial


def get_dataset(opts, train=True):
    """ Dataset And Augmentation
    """
    TRAIN_CV = 0.8

    if opts.dataset == 'voc':
        dataset = VOCSegmentation
        train_transform = Compose([
            RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
        val_transform = Compose([
            PadCenterCrop(size=opts.crop_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
        # no crop, batch size = 1
        test_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])

    elif opts.dataset == 'cts':
        
        dataset = partial(Cityscapes, cl19=True)
        
        if opts.model == 'bisenetv2':
            train_transform = Compose([
                RandomHorizontalFlip(0.5),
#                 ---PICK ONE---
#                 Resize((512, 1024)),
                RandomResizeRandomCrop(),
#                 RandomResizedCrop((512, 1024), scale=(.5, 1), ratio=(2., 2.)),
#                 --------------
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])
            val_transform = Compose([
#                 ---PICK ONE---
                Resize((512, 1024)),
#                 RandomResizeRandomCrop(),
#                 RandomResizedCrop((512, 1024), scale=(.5, 1), ratio=(2., 2.)),
#                 --------------
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])
            test_transform = Compose([
#                 ---PICK ONE---
                Resize((512, 1024)),
#                 RandomResizeRandomCrop(),
#                 RandomResizedCrop((512, 1024), scale=(.5, 1), ratio=(2., 2.)),
#                 --------------
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])
            print(train_transform)
            print(val_transform)
            print(test_transform)
        else:
            train_transform = Compose([
                RandomScale((0.7, 2)),  # Using RRC should be (0.25, 0.75)
                RandomCrop(opts.crop_size),  # set 512
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])    
            val_transform = Compose([
                PadCenterCrop(size=opts.crop_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])
            test_transform = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])
    else:
        raise NotImplementedError

    if train:
        if opts.cross_val:
            train_dst = dataset(root=opts.data_root, train=True, transform=None)
            train_len = int(TRAIN_CV * len(train_dst))
            idx = list(range(len(train_dst)))
            random.shuffle(idx)
            train_dst = Subset(train_dst, idx[:train_len], train_transform)
            val_dst = Subset(train_dst, idx[train_len:], val_transform)
        else:
            train_dst = dataset(root=opts.data_root, train=True, transform=train_transform)
            if opts.model == 'bisenetv2':
                val_dst = dataset(root=opts.data_root, train=False, transform=val_transform, test_bisenetv2=True)
            else:
                val_dst = dataset(root=opts.data_root, train=False, transform=val_transform)

        return train_dst, val_dst
    else:
        if opts.model == 'bisenetv2':
            test = dataset(root=opts.data_root, train=False, transform=test_transform, test_bisenetv2=True)
        else:
            test = dataset(root=opts.data_root, train=False, transform=test_transform)
        return test
