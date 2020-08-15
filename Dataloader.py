from PIL import Image
from utils import transforms as T
import torch
import torch.utils.data
import json
import os
import numpy as np


# the dataset loader is of two version, one for 2-class, another for 3-calss seg.
class ISBI14Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        # following two lines are used for 2-class seg. with cer93 aug.
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'train_3.21', 'trainset'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'train_3.21', 'train_Cytoplasm_masks'))))

        # following three lines for 3-class seg. without cer93 aug.
        # self.imgs = list(sorted(os.listdir(os.path.join(root, 'test900', 'test900_Images'))))
        # self.cytoplasms = list(sorted(os.listdir(os.path.join(root, 'test900', 'testMasks'))))
        # self.nucleis = list(sorted(os.listdir(os.path.join(root, 'test900', 'test_Nuclei'))))

    def __getitem__(self, idx):
        """
        these codes are for 2-classes segmentation problem, which loads only cytoplasms as masks
        """
        # load images ad masks
        img_path = os.path.join(self.root, 'train_3.21', 'trainset', self.imgs[idx])
        mask_path = os.path.join(self.root, 'train_3.21', 'train_Cytoplasm_masks', self.masks[idx])
        single_masks_path = list(sorted(os.listdir(mask_path)))
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        masks = []
        for single_mask_path in single_masks_path:
            mask = Image.open(os.path.join(mask_path, single_mask_path))
            mask = np.array(mask)
            masks.append(mask)

        num_objs = len(single_masks_path)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        # # the following codes are for 3-classes segmentation problem, which loads cytoplasms and nucleis
        # # load images
        # img_path = os.path.join(self.root, 'test900', 'test900_Images', self.imgs[idx])
        # img = Image.open(img_path).convert("RGB")
        # cytoplasms_path = os.path.join(self.root, 'test900', 'testMasks', self.cytoplasms[idx])
        # nucleis_path = os.path.join(self.root, 'test900', 'test_Nuclei', self.nucleis[idx])
        # single_cytoplasms_path = list(sorted(os.listdir(cytoplasms_path)))
        # masks = []
        # for single_cytoplasm_path in single_cytoplasms_path:
        #     mask = Image.open(os.path.join(cytoplasms_path, single_cytoplasm_path))
        #     mask = np.array(mask)
        #     masks.append(mask)
        #
        # # create labels for cytoplasm
        # labels = [1]*len(masks)
        #
        # # load nuclei 2-bit images
        # nuclei_img = np.asarray(Image.open(nucleis_path))
        #
        # # convert semantic labels to instances labels
        # # for the reason that almost none overlapping areas among nucleis, the instances can be
        # # easily handled with the following method, which belongs to images processing problem
        # nucleis = measure.label(nuclei_img,connectivity=2)  #8连通区域标记
        #
        # # instances are encoded as different colors
        # nuclei_ids = np.unique(nucleis)
        #
        # # first id is the background, so remove it
        # nuclei_ids = nuclei_ids[1:]
        #
        # # split the color-encoded mask into a set of binary masks
        # nuclei_masks = nucleis == nuclei_ids[:, None, None]
        #
        # # create labels for nucleis
        # nuclei_labels = [2]*len(nuclei_ids)
        #
        # # put cytoplasms and nucleis masks together
        # masks.extend(nuclei_masks)
        # labels.extend(nuclei_labels)
        # num_objs = len(masks)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])
        #
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # # there is only one class
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Cervix93Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.jsons = list(sorted(os.listdir(os.path.join(root, "JSONFiles"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        json_path = os.path.join(self.root, "JSONFiles", self.jsons[idx])
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        single_mask_path = list(sorted(os.listdir(mask_path)))
        img = Image.open(img_path).convert("RGB")
        masks = []
        for path in single_mask_path:
            mask = Image.open(os.path.join(mask_path, path))
            mask = np.array(mask)
            masks.append(mask)

        # get the class labels of the cells, that is, 'H', 'L', or 'N'
        data = json.load(open(json_path))
        shapes = data['shapes']
        class_name_to_value = {'N': 1, 'L': 2, 'H': 3}
        cls_id = []
        for shape in shapes:
            label_name = shape['label']
            cls_name = label_name[0]
            cls_id.append(class_name_to_value[cls_name])

        # get bounding box coordinates for each mask
        num_objs = len(single_mask_path)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class except for the background
        labels = torch.as_tensor(cls_id, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class cer_aug_dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Aug_imgs"))))
        # self.jsons = list(sorted(os.listdir(os.path.join(root, "JSONFiles"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Aug_masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "Aug_imgs", self.imgs[idx])
        # json_path = os.path.join(self.root, "JSONFiles", self.jsons[idx])
        mask_path = os.path.join(self.root, "Aug_masks", self.masks[idx])
        single_mask_path = list(sorted(os.listdir(mask_path)))
        img = Image.open(img_path).convert("RGB")
        masks = []
        for path in single_mask_path:
            mask = Image.open(os.path.join(mask_path, path))
            mask = np.array(mask)
            masks.append(mask)

        # get the class labels of the cells, that is, 'H', 'L', or 'N'
        # data = json.load(open(json_path))
        # shapes = data['shapes']
        # class_name_to_value = {'N': 1, 'L': 2, 'H': 3}
        # cls_id = []
        # for shape in shapes:
        #     label_name = shape['label']
        #     cls_name = label_name[0]
        #     cls_id.append(class_name_to_value[cls_name])

        # get bounding box coordinates for each mask
        num_objs = len(single_mask_path)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class except for the background
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = [T.ToTensor()]
    # converts the image, a PIL image, into a PyTorch Tensor
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)