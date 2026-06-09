import torch
import torch.nn.functional as F
from utils import box_ops

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_src_permutation_idx(indices):
    # permute predictions following indices
    # [num_all_gt]  记录每个gt都是来自哪张图片的 idx
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    # 记录匹配到的预测框的idx
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def loss_labels(outputs, targets, indices, num_class):
    """
    :param outputs: 模型预测输出
    :param targets: Ground Truth标签
    :param indices: matcher()输出
    :param num_class: 真实类别的数量
    :return: 类别损失 交叉熵损失
    """
    src_logits = outputs['pred_logits']  # 分类：[bs, 50, 1类别]

    # idx tuple:2  0=[num_all_gt] 记录每个gt属于哪张图片  1=[num_all_gt] 记录每个匹配到的预测框的index
    idx = get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2], 0,  # 这里设置为0，固定id=0为背景类，DETR中设置的背景类为num_class
                                dtype=torch.int64, device=src_logits.device)  # 这里生成的target_classes值都初始化为0背景类
    # 正样本+负样本  上面匹配到的预测框作为正样本
    target_classes[idx] = target_classes_o

    empty_weight = torch.ones(num_class+1).to(device)  # 权重系数，这里num_class+1表示给几个分类分配一下权重，默认为1
    empty_weight[0] = 0.1  # 对于背景类，权重设置为0.1，避免背景过度影响

    # 分类损失 = 正样本 + 负样本
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)

    # losses: 'loss_ce': 分类损失 固定权重
    return 5*loss_ce


def loss_boxes(outputs, targets, indices):
    """
    :param outputs: 模型预测输出
    :param targets: Ground Truth标签
    :param indices: matcher()输出
    :return: 框box回归损失 L1 Loss
    """
    num_boxes = sum(len(t["labels"]) for t in targets)
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    # 记录每个匹配到的预测框的index
    idx = get_src_permutation_idx(indices)

    # 这个batch的所有正样本的预测框坐标
    src_boxes = outputs['pred_boxes'][idx]
    # 这个batch的所有gt框坐标
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    # 计算L1损失
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    l1_loss = loss_bbox.sum() / num_boxes

    # 计算GIOU损失
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(target_boxes)))

    giou_loss = loss_giou.sum() / num_boxes

    # 'loss_bbox': L1回归损失  'loss_giou': giou回归损失，固定损失权重分别为5、2
    return 3*l1_loss + 6*giou_loss


def loss_masks(outputs, targets, indices):
    """
    :param outputs: 模型预测输出
    :param targets: Ground Truth标签
    :param indices: 二分匹配的输出match_output
    :return: 总损失（Dice损失和CE Loss加权求和）
    """
    num_class = 1  # 这里固定类别数只有一个病灶类，待改进代码

    # 通过indices获得对应预测的索引
    idx = get_src_permutation_idx(indices)

    # 选出匹配到的预测掩码 [num_matches, 2, W, H]
    src_masks_logits = outputs['pred_masks'][idx]
    # 对 logits 进行 sigmoid 得到预测概率
    src_masks = src_masks_logits.sigmoid()

    # 组合目标掩码
    targets_masks = torch.cat([t["masks"][j] for t, (_, j) in zip(targets, indices)], dim=0)

    # 定义类别权重：背景类别（索引0）0.1，目标类别（索引1）1.0
    class_weights = torch.tensor([0.1, 1.0], device=src_masks_logits.device)

    # 计算 Dice Loss（按类别计算，然后加权求和）
    smooth = 1e-6
    dice_loss = 0.0
    # 遍历类别，假设固定类别数为2（num_class + 背景类）
    for i in range(num_class + 1):
        # 取出第 i 个类别的预测和目标掩码，形状为 [num_matches, W, H]
        pred = src_masks[:, i, :, :]
        target = targets_masks[:, i, :, :]
        # 计算每个样本的交集
        intersection = (pred * target).sum(dim=(1, 2))
        # 计算每个样本的 Dice 损失
        dice_loss_channel = 1 - (2. * intersection + smooth) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth)
        # 用该类别的权重乘后取平均
        dice_loss += class_weights[i] * dice_loss_channel.mean()

    # 计算 CE Loss，利用 weight 参数为每个类别加权
    # 注意：F.binary_cross_entropy_with_logits 能自动处理形状
    ce_loss = F.binary_cross_entropy_with_logits(
        src_masks_logits,  # logits 输入
        targets_masks,     # 目标掩码
        weight=class_weights.view(1, num_class + 1, 1, 1),  # 重塑权重使其能广播到 [N,2,W,H]
        reduction='mean'
    )

    # 'masks_loss': Dice损失 固定权重
    return 6*ce_loss + 4*dice_loss
