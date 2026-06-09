import json
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class DataSet(torch.utils.data.Dataset):
    def __init__(self, input_dir):
        """
        :param input_dir: 输入数据文件路径
        """
        self.images_dir = os.path.join(input_dir, 'images')
        self.text_dir = os.path.join(input_dir, 'texts')
        self.masks_dir = os.path.join(input_dir, 'masks')
        self.boxes_dir = os.path.join(input_dir, 'boxes')

        self.img_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Grayscale(),  # 保持mask为单通道
            transforms.ToTensor()
        ])

        self.images_names = [os.path.splitext(f)[0] for f in sorted(os.listdir(self.images_dir))]

    def __len__(self):
        # 返回数据集的大小
        return len(self.images_names)

    def __getitem__(self, idx):
        # 加载图像和mask
        image_name = self.images_names[idx]
        image_path = os.path.join(self.images_dir, image_name + '.png')
        text_path = os.path.join(self.text_dir, image_name + '.txt')
        mask_path = os.path.join(self.masks_dir, image_name + '.png')
        box_path = os.path.join(self.boxes_dir, image_name + '.json')

        image = Image.open(image_path).convert('RGB')
        w, h = image.size

        with open(text_path, 'r') as file:
            text = file.read()

        mask = Image.open(mask_path)

        with open(box_path, 'r') as box_file:
            box_file = json.load(box_file)

        # 应用变换transform
        image = self.img_transform(image)

        mask = self.mask_transform(mask)
        background_mask = 1 - mask

        mask = torch.cat([background_mask, mask], dim=0)

        # 处理box格式：变为中心点与宽高+归一化
        box, label = prepare_box(box_file, h, w)

        mask = mask.expand(len(label), -1, -1, -1)  # 扩展mask掩码

        target = {
            "name": box_file["name"],
            "labels": label,
            "boxes": box,
            "masks": mask
        }

        return image, text, target


def prepare_box(box_file, height, width):
    """
    :param box_file: json文件
    :param height: 图像的高
    :param width 图像的宽
    """
    # 提取boxes和labels字段
    boxes = box_file["boxes"]
    labels = box_file["labels"]

    # 转换boxes字段
    converted_boxes = []
    for box in boxes:
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2
        cx_normalized = cx / width
        cy_normalized = cy / height
        w_normalized = w / width
        h_normalized = h / height
        converted_boxes.append([cx_normalized, cy_normalized, w_normalized, h_normalized])

    # 将转换后的boxes和labels转换为tensor格式
    boxes_tensor = torch.tensor(converted_boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    return boxes_tensor, labels_tensor
