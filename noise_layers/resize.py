import torch.nn as nn
import torch.nn.functional as F
from noise_layers.crop import random_float
import torch


class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method

    def forward(self, noised_and_cover):
        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover[0]
        
        # 保存原始尺寸
        original_size = noised_image.shape[2:]  # 获取H,W
        
        # 进行缩放
        resized = F.interpolate(
            noised_image,
            scale_factor=(resize_ratio, resize_ratio),
            mode=self.interpolation_method)
        
        # 创建白色背景 (假设图像值范围是[0,1])
        white_background = torch.ones_like(noised_image)
        # 如果图像值范围是[0,255]，使用：
        # white_background = torch.ones_like(noised_image) * 255.0
        
        # 计算放置缩放后图像的位置（居中）
        h_resized, w_resized = resized.shape[2], resized.shape[3]
        h_start = (original_size[0] - h_resized) // 2
        w_start = (original_size[1] - w_resized) // 2
        
        # 确保不会越界
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        h_end = min(original_size[0], h_start + h_resized)
        w_end = min(original_size[1], w_start + w_resized)
        
        # 调整resized图像的大小以防超出边界
        resized_h = h_end - h_start
        resized_w = w_end - w_start
        if resized_h != h_resized or resized_w != w_resized:
            resized = resized[:, :, :resized_h, :resized_w]
        
        # 将缩放后的图像放置在白色背景上
        white_background[:, :, h_start:h_end, w_start:w_end] = resized
        
        noised_and_cover[0] = white_background

        return noised_and_cover
