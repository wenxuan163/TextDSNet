import os

import clip
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

import torch
from torch import optim
import argparse

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from utils.log import log

from datasets.dataset import DataSet

from utils.collate_fn import collate_fn
from models import TextDSNet
from models import HungarianMatcher
from loss import loss_labels, loss_boxes, loss_masks

from torch.utils.tensorboard import SummaryWriter

"""
    自定义args参数
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='Batch size of train dataloader.')
parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, help='Learning rate of training.')
parser.add_argument('--output', type=str, default='output_test', help='where to save the model.')
parser.add_argument('--train_path', type=str, default='ISLES2022', help='train path.')
parser.add_argument('--num_class', type=int, default=1, help='number of true class.')
args = parser.parse_args()

# 创建一个SummaryWriter的实例
writer = SummaryWriter(log_dir=os.path.join(args.output, 'runs'))

"""
    配置日志
"""
logger = log(args.output)

logger.info(f"Loss设置：label中ce权重5，box中l1,giou权重为3,6，mask中ce,dice为6,4。")

"""
    使用设备Device
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备device：{device}\n")

''' 原始的Dataset '''
dataset_train = DataSet(input_dir=args.train_path)
data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                collate_fn=collate_fn)

network = TextDSNet(in_chans=3).to(device)
logger.info("Load model success.")

# # 加载预训练CLIP模型
# clip_model, _ = clip.load('ViT-B/32', device=device, jit=False)
# # 冻结CLIP模型参数,训练时CLIP不参与更新
# for param in clip_model.parameters():
#     param.requires_grad = False

# 加载MedClip模型
clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
clip_model.from_pretrained()
clip_model.to(device)
for param in clip_model.parameters():
    param.requires_grad = False
processor = MedCLIPProcessor()
logger.info("Load clip success.")

# 创建Adam优化器，并放入模型的参数
optimizer = optim.Adam(network.parameters(), lr=args.lr)
logger.info(f"使用优化器：{optimizer}")

# 自适应学习率调度器
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
logger.info(f"使用学习率调整策略：{scheduler}")

# 二分匹配算法(匈牙利匹配算法)
matcher = HungarianMatcher()

# 如果不存在输出文件夹，则创建文件夹
if not os.path.exists(args.output):
    os.makedirs(args.output)

logger.info("Start Training.")
for epoch in range(args.epochs):
    # 训练模式
    network.train()
    logger.info(f"\n\nEpoch [{epoch}]:\n")
    for batch_index, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train)):
        img_data = data[0].to(device)  # 下标0返回image
        text_data = data[1]  # 下标1返回text
        target_data = [{k: v.to(device) for k, v in t.items()} for t in data[2]]  # data下标2返回target

        output = network(img_data, text_data, clip_model, processor, device)

        match_output = matcher(outputs=output, targets=target_data)

        labels_loss = loss_labels(output, target_data, match_output, num_class=args.num_class)
        boxes_loss = loss_boxes(output, target_data, match_output)
        masks_loss = loss_masks(output, target_data, match_output)

        # 使用时变Uncertainty加权
        process_time = epoch / args.epochs
        reg_label = process_time * network.log_var_labels  # 时变正则化项
        reg_box = process_time * network.log_var_boxes
        reg_mask = process_time * network.log_var_masks

        loss = (
                torch.exp(-network.log_var_labels) * labels_loss +
                torch.exp(-network.log_var_boxes) * boxes_loss +
                torch.exp(-network.log_var_masks) * masks_loss +
                0.5 * (network.log_var_boxes + network.log_var_labels + network.log_var_masks) +
                reg_label + reg_box + reg_mask
        )

        if batch_index % 5 == 0:
            logger.info(
                f"\nbatch index: [{batch_index} / {len(data_loader_train)}]: \nlabel损失值: {labels_loss.item()}, "
                f"boxes损失值: {boxes_loss.item()}, masks损失值: {masks_loss.item()}, Total Loss: {loss.item()}")

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播得到参数梯度
        optimizer.step()  # 更新参数

    writer.add_scalars('Train Loss/multi', {'label loss': labels_loss, 'box loss': boxes_loss, 'mask loss': masks_loss},
                       epoch)
    writer.add_scalar('Train Loss/sigma_σ²_label', torch.exp(network.log_var_labels) - 1, epoch)
    writer.add_scalar('Train Loss/sigma_σ²_box', torch.exp(network.log_var_boxes) - 1, epoch)
    writer.add_scalar('Train Loss/sigma_σ²_mask', torch.exp(network.log_var_masks) - 1, epoch)
    writer.add_scalar('Train Loss/total', loss, epoch)  # 调用可视化方法

    # 保存本次epoch训练后的模型
    torch.save(network.state_dict(), f'{args.output}/model_epoch_{epoch}.pth')
    logger.info(f"模型已保存: {args.output}/model_epoch_{epoch}.pth")

    scheduler.step()
    logger.info(f"使用学习率lr = {optimizer.param_groups[0]['lr']}")
    writer.add_scalar('Learning Rate/train', optimizer.param_groups[0]['lr'], epoch)  # 调用可视化方法

writer.close()  # 关闭训练可视化

# 保存训练的最后一次模型
torch.save(network.state_dict(), f'{args.output}/last.pth')
logger.info(f"last.pth模型已保存: {args.output}/last.pth")
