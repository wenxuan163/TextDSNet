import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse

import clip

from .segnet import SegNet
from .detnet import DetNet

# from segnet import SegNet
# from detnet import DetNet

# Box MLP
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 多头交叉注意力计算模块 (跨模态)
class CrossAttention(nn.Module):

    def __init__(self,
                 embed_dim,  # 输入token的dim
                 num_heads,
                 qk_scale=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 根号d，缩放因子

        # 一个线性层得到的特征维度，由多个头分别关注某一小块维度的特征
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x1, x2):
        # [batch_size, Length, Dim]
        x1_B, x1_L, x1_D = x1.shape
        x2_B, x2_L, x2_D = x2.shape

        # Q: [batch_size, Length, Dim] reshape=> [batch_size, Length, num_heads, Dim_each_num_heads]
        #    permute=> [batch_size, num_heads, Length, Dim_each_num_heads]
        # K: [batch_size, Length, Dim] reshape=> [batch_size, Length, num_heads, Dim_each_num_heads]
        #    permute=> [batch_size, num_heads, Dim_each_num_heads, Length]
        # 如此 Q 才可与 K进行矩阵乘法
        q = self.proj_q(x1).reshape(x1_B, x1_L, self.num_heads, x1_D // self.num_heads).permute(0, 2, 1, 3)
        k = self.proj_k(x2).reshape(x2_B, x2_L, self.num_heads, x2_D // self.num_heads).permute(0, 2, 3, 1)
        v = self.proj_v(x2).reshape(x2_B, x2_L, self.num_heads, x2_D // self.num_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(q, k) * self.scale
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, v).transpose(1, 2).reshape(x1_B, x1_L, x1_D)

        return output


class TextDSNet(nn.Module):

    def __init__(self, in_chans=3, img_size=224, patch_size=2, embed_dim=48, num_heads=[3, 6, 12, 24],
                 total_layers=4, num_queries=20, queries_dim=256, num_class=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.total_layers = total_layers
        self.num_class = num_class

        self.segnet = SegNet(in_chans=in_chans, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                             num_heads=num_heads, total_layers=total_layers)
        self.segdet_cross_text_proj = nn.Linear(512, 256)  # 低级语义特征图对应的text_proj
        self.segdet_cross_img_proj = nn.Linear(48, 256)  # 低级语义特征图的proj
        self.segdet_cross_attn = CrossAttention(embed_dim=256, num_heads=8)
        self.segdet_cross_norm = nn.LayerNorm(256)
        self.query_embed = nn.Embedding(num_queries, queries_dim)  # query默认预测固定数量个
        # self.query_proj = nn.Linear(512, queries_dim)  # 将clip的文本特征维度降维到query维度
        self.detnet = DetNet(img_size=img_size, embed_dim=embed_dim, total_layers=total_layers)
        # self.class_embed = nn.Linear(queries_dim, num_class + 1)  # 默认输入class为2：检测分割0背景类，1脑卒中类
        self.class_embed = nn.Sequential(
            nn.Linear(queries_dim, queries_dim),
            nn.ReLU(),
            nn.Linear(queries_dim, num_class + 1)
        )
        self.bbox_embed = MLP(queries_dim, queries_dim, output_dim=4)  # out_dim为4即box坐标点[x,y,w,h]
        self.seg_head = nn.Conv2d(queries_dim, queries_dim, kernel_size=1)  # 将分割特征图维度对齐query embed

        # 最终分割图输出
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=num_class + 1, kernel_size=1)

        # 定义原始可训练参数（无约束）
        self.raw_labels = nn.Parameter(torch.zeros(1))
        self.raw_boxes = nn.Parameter(torch.zeros(1))
        self.raw_masks = nn.Parameter(torch.zeros(1))

        # 通过属性访问安全的 log(1 + α²)

    @property
    def log_var_labels(self):
        # 计算 α² = exp(raw_labels)
        alpha_sq = torch.exp(self.raw_labels)
        # 返回 log(1 + α²)
        return torch.log(1 + alpha_sq)

    @property
    def log_var_boxes(self):
        alpha_sq = torch.exp(self.raw_boxes)
        return torch.log(1 + alpha_sq)

    @property
    def log_var_masks(self):
        alpha_sq = torch.exp(self.raw_masks)
        return torch.log(1 + alpha_sq)

    def forward(self, image_features, texts, clip_model, processor, device):
        """ 使用Clip """
        # texts = clip.tokenize(texts=texts).to(device)  # 编码texts
        # with torch.no_grad():
        #     text_features = clip_model.encode_text(texts).to(
        #         dtype=torch.float32)  # CLIP Text Encoder后的文本特征类型是float16，应变为torch默认的float32

        """ 使用MedClip """
        # 使用 MedCLIPProcessor 来 tokenize 文本
        texts = processor(text=texts, return_tensors="pt", padding=True)
        # 移除不需要的 token_type_ids 键
        if "token_type_ids" in texts:
            texts.pop("token_type_ids")
        # 将输入移到对应设备上
        texts = {k: v.to(device) for k, v in texts.items()}
        with torch.no_grad():
            text_features = clip_model.encode_text(**texts).to(dtype=torch.float32)

        #  得到分割网络的多尺度特征图
        *segnet_high, segnet_low = self.segnet(image_features, text_features)


        # 检测模型输出
        detnet_out = self.detnet(self.query_embed.weight, text_features, *segnet_high)

        outputs_class = self.class_embed(detnet_out).permute(1, 0, 2)
        outputs_coord = self.bbox_embed(detnet_out).sigmoid().permute(1, 0, 2)

        # 分割掩码输出
        query_seg = detnet_out.permute(1, 0, 2)

        # 为低级语义特征图进行跨模态交叉注意力（高分辨率）
        reshape_text = text_features.unsqueeze(1)
        proj_text_features = self.segdet_cross_text_proj(reshape_text)

        B, C, H, W = segnet_low.size()
        reshaped_segnet_low = segnet_low.reshape(B, H * W, C)
        proj_segnet_low = self.segdet_cross_img_proj(reshaped_segnet_low)
        cross_segnet_low = self.segdet_cross_norm(proj_segnet_low +
                                                  self.segdet_cross_attn(proj_segnet_low, proj_text_features))

        cross_segnet_low = cross_segnet_low.reshape(B, -1, H, W)
        pixel_seg = self.seg_head(cross_segnet_low)

        pred_mask = torch.einsum('bql,blhw->bqhw', query_seg, pixel_seg)
        b, q, _, _ = pred_mask.shape

        pred_mask = F.interpolate(pred_mask, scale_factor=2, mode='bilinear')  # 双线性插值上采样为原始分辨率

        pred_mask = pred_mask.reshape(-1, 1, self.img_size, self.img_size)

        outputs_mask = self.final_conv(pred_mask)
        # outputs_mask = outputs_mask.reshape(b, q, self.num_class + 1, self.img_size, self.img_size).sigmoid()
        outputs_mask = outputs_mask.reshape(b, q, self.num_class + 1, self.img_size, self.img_size)

        output = {
            "pred_masks": outputs_mask,
            # "pred_masks": outputs_mask,
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,  # 输出框格式为[x,y,w,h]，而且都是归一化的结果
        }

        return output
