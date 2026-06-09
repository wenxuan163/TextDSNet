import torch
import torch.nn as nn
from math import factorial
from itertools import permutations

device = "cuda" if torch.cuda.is_available() else "cpu"


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


# 多头交叉注意力计算模块 (多尺度特征)
class CrossAttentionMulti(nn.Module):

    def __init__(self,
                 embed_dim,  # 输入token的dim
                 num_heads,
                 qk_scale=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 根号d，缩放因子

    def forward(self, q, k, v):
        # [batch_size, Length, Dim]
        q_B, q_L, q_D = q.shape
        k_B, k_L, k_D = k.shape
        v_B, v_L, v_D = v.shape

        # Q: [batch_size, Length, Dim] reshape=> [batch_size, Length, num_heads, Dim_each_num_heads]
        #    permute=> [batch_size, num_heads, Length, Dim_each_num_heads]
        # K: [batch_size, Length, Dim] reshape=> [batch_size, Length, num_heads, Dim_each_num_heads]
        #    permute=> [batch_size, num_heads, Dim_each_num_heads, Length]
        # 如此 Q 才可与 K进行矩阵乘法
        q = q.reshape(q_B, q_L, self.num_heads, q_D // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(k_B, k_L, self.num_heads, k_D // self.num_heads).permute(0, 2, 3, 1)
        v = v.reshape(v_B, v_L, self.num_heads, v_D // self.num_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(q, k) * self.scale
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, v).transpose(1, 2).reshape(q_B, q_L, q_D)

        return output


class FFN(nn.Module):

    def __init__(self, in_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2 * in_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(2 * in_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerDecoderBlock(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads)
        self.multi_head_attn = nn.MultiheadAttention(dim, num_heads)
        self.ffn = FFN(in_features=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_embed):
        """
        :param tgt: 上一层解码器的解码输出，第一层的tgt=torch.zeros_like(query_embed) 为零矩阵
        :param memory: segnet对decoder的输出
        :param query_embed: 学习的object query
        """

        q = k = self.with_pos_embed(tgt, query_embed)  # Self-Attention部分的QK
        tgt2 = self.self_attn(query=q, key=k, value=tgt)[0]  # 自注意力计算
        tgt = tgt + tgt2  # 残差连接
        tgt = self.norm1(tgt)  # 层归一化
        tgt2 = self.multi_head_attn(query=self.with_pos_embed(tgt, query_embed),
                                    key=memory,
                                    value=memory)[0]  # 多头交叉注意力计算
        tgt = tgt + tgt2  # 残次连接
        tgt = self.norm2(tgt)  # 层归一化
        tgt2 = self.ffn(tgt)  # feed and forward
        tgt = tgt + tgt2  # 残次连接
        tgt = self.norm3(tgt)  # normal归一化

        return tgt


class TransformerDecoder(nn.Module):

    def __init__(self, dim, num_heads, num_layer):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerDecoderBlock(dim=dim, num_heads=num_heads)
                                     for _ in range(num_layer)])

    def forward(self, memory, query_embed):
        """
        :param memory: 输入至Transformer Decoder的输入  [Batch_size, L, Dim]
        :param query_embed: Decoder预测的可学习的query embed
        """
        bs, l, d = memory.shape
        memory = memory.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # tgt初始化，意义为初始化需要预测的目标。因为一开始不清楚需要什么样的目标，所以初始化为0，它会在decoder中不断被refine，
        # 但真正在学习的是query_embed, 学习到的是整个数据集中目标物体的统计特征。而tgt在每一个epoch都会初始化。
        tgt = torch.zeros_like(query_embed)

        # 获取decoder输出
        for decoder_block in self.blocks:
            tgt = decoder_block(tgt, memory, query_embed)

        return tgt


class DetNet(nn.Module):

    def __init__(self, img_size=224, embed_dim=48, total_layers=4):
        super().__init__()
        self.img_size = img_size
        self.total_layers = total_layers

        '''
            跨模态交叉注意力部分
        '''
        # 将clip输出的text dim变为256(Clip Text Encoder输出dim为512)
        self.clip_text_proj = nn.ModuleList([nn.Linear(512, 256)
                                             for _ in range(total_layers - 1)])
        # 将segnet输出的多尺度特征图dim变为256(Clip Text Encoder和img都统一输出dim为256)
        self.clip_img_proj = nn.ModuleList([nn.Linear(int(embed_dim * 2 ** (i + 1)), 256)
                                            for i in range(total_layers - 1)[::-1]])
        self.clip_cross_attn = nn.ModuleList([CrossAttention(embed_dim=256, num_heads=8)
                                              for _ in range(total_layers - 1)])
        self.clip_norm = nn.ModuleList([nn.LayerNorm(256)
                                        for _ in range(total_layers - 1)])
        '''
            多尺度特征图交叉注意力部分
        '''
        # self.multi_proj = nn.ModuleList([nn.Linear(512, 256)
        #                                  for _ in range(total_layers - 1)])
        self.proj_q = nn.ModuleList([nn.Linear(256, 256, bias=False)
                                     for _ in range(total_layers - 1)])
        self.proj_k = nn.ModuleList([nn.Linear(256, 256, bias=False)
                                     for _ in range(total_layers - 1)])
        self.proj_v = nn.ModuleList([nn.Linear(256, 256, bias=False)
                                     for _ in range(total_layers - 1)])
        self.multi_cross_attn = nn.ModuleList([CrossAttentionMulti(embed_dim=256, num_heads=8)
                                               for _ in range(factorial(total_layers - 1))])
        self.cross_attn_norm = nn.ModuleList([nn.LayerNorm(256)
                                              for _ in range(factorial(total_layers - 1))])
        self.multi_mlp = nn.ModuleList([FFN(in_features=256)
                                        for _ in range(factorial(total_layers - 1))])
        self.multi_norm = nn.ModuleList([nn.LayerNorm(256)
                                         for _ in range(total_layers - 1)])
        # self.fuse_linear = nn.Linear(256, 96)
        '''
            Transformer Decoder部分（DETR Decoder）
        '''
        # self.transformer_decoder = TransformerDecoder(dim=96, num_heads=8, num_layer=6)
        self.transformer_decoder = TransformerDecoder(dim=256, num_heads=8, num_layer=6)

    def forward(self, query_embed, text_features, *segnet_high):
        # print('query_embed: ', query_embed.size())
        """
        :param query_embed: 检测网络部分输入的object query
        :param text_features: 分割网络与检测网络中间跨模态注意力的文本features
        :param segnet_high: 分割网络输出的若干高级语义特征图
        """

        ''' 跨模态交叉注意力部分 '''

        clip_attn = []  # 存放跨模态交叉注意力计算的结果
        for i in range(len(segnet_high)):

            B, C, H, W = segnet_high[i].size()
            reshape_segnet_high = segnet_high[i].reshape(B, H * W, C)
            reshape_segnet_high = self.clip_img_proj[i](reshape_segnet_high)
            reshape_text = text_features.unsqueeze(1)
            reshape_text = self.clip_text_proj[i](reshape_text)

            # 计算时，Q为视觉，K V为文本,计算完图像对文本的注意后，将图像特征图 + 注意力特征图
            clip_cross_attn = self.clip_norm[i](reshape_segnet_high +
                                                self.clip_cross_attn[i](reshape_segnet_high, reshape_text))
            # 计算完跨模态注意力后要经过multi_proj层降维 512 -> 256
            clip_attn.append(clip_cross_attn)

        ''' 多尺度特征图交叉注意力部分 '''
        proj_q = []  # 存放多尺度特征图所各自产生的Q
        proj_k = []
        proj_v = []
        for i in range(self.total_layers - 1):
            proj_q.append(self.proj_q[i](clip_attn[i]))
            proj_k.append(self.proj_k[i](clip_attn[i]))
            proj_v.append(self.proj_v[i](clip_attn[i]))

        array = list(range(self.total_layers - 1))  # [0,1,2]，两元素排列组合[0,1][0,2][1,0][1,2][2,0][2,1]

        multi_attn = [[] for _ in range(len(clip_attn))]  # 存放多尺度特征图交叉注意力的结果，[a][b]表示a对b的交叉注意力
        sum_attn = []  # 存放每个尺度的特征图通过多尺度交叉注意力计算的结果
        group_index = 0  # 存放相同i的注意力
        group_counter = 0  # 计数器
        for i, (a, b) in enumerate(permutations(array, 2)):
            # 每取出两个元素（[a][0] 和 [a][1]）则group_index++
            if group_counter == 2:
                group_index += 1
                group_counter = 0

            multi_attn_mlp = self.multi_mlp[i](
                self.cross_attn_norm[i](clip_attn[a] + self.multi_cross_attn[i](proj_q[a], proj_k[b], proj_v[b])))
            multi_attn[group_index].append(multi_attn_mlp)

            group_counter += 1

        for i, sum_items in enumerate(multi_attn):
            item1, item2 = sum_items
            sum_attn.append(self.multi_norm[i](item1 + item2))

        combined_result = torch.cat(sum_attn, dim=1)

        ''' Transformer Decoder部分（DETR Decoder） '''

        # decoder = self.transformer_decoder(fuse_result, query_embed)
        # decoder输出 [query个数, batch_size, query_dim维度]
        decoder = self.transformer_decoder(combined_result, query_embed)

        return decoder
