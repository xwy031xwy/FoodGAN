# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738
import random

import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', random_choice=True):
    if policy:
        policies = policy.split(',')
        if random_choice:
            policies = [random.choice(policies)]
        for p in policies:
            for f in AUGMENT_FNS[p]:
                x = f(x)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.2):
    print(x.shape)
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    print(x.shape)
    return x


def manual_mirror_padding(image, pad_left, pad_right, pad_top, pad_bottom):
    # Left padding
    if pad_left != 0:
        left_pad = image[:, :, :, :pad_left.item()].flip(dims=[3])
        image = torch.cat([left_pad, image], dim=3)

    # Right padding
    if pad_right != 0:
        right_pad = image[:, :, :, -pad_right.item():].flip(dims=[3])
        image = torch.cat([image, right_pad], dim=3)

    # Top padding
    if pad_top != 0:
        top_pad = image[:, :, :pad_top.item(), :].flip(dims=[2])
        image = torch.cat([top_pad, image], dim=2)

    # Bottom padding
    if pad_bottom != 0:
        bottom_pad = image[:, :, -pad_bottom.item():, :].flip(dims=[2])
        image = torch.cat([image, bottom_pad], dim=2)

    return image


def rand_translation_with_mirror(x, ratio=0.2):
    assert len(x.shape) == 4, 'Input tensor must be 4D'

    batchsize, channels, size, _ = x.shape
    max_shift = int(x.size(2) * ratio + 0.5)
    # 生成随机方向和随机距离
    direction = random.choice(['left', 'right', 'up', 'down'])
    shift = torch.randint(1, max_shift + 1, (1,)).item()

    # 将输入张量的边缘进行镜像填充，为后续的平移操作做准备
    padded_tensor = F.pad(x, (max_shift, max_shift, max_shift, max_shift), mode='reflect')

    # 根据方向和平移量计算裁剪的索引
    if direction == 'left':
        start_x, end_x, start_y, end_y = max_shift, size + max_shift, max_shift + shift, size + max_shift + shift
    elif direction == 'right':
        start_x, end_x, start_y, end_y = max_shift, size + max_shift, max_shift - shift, size + max_shift - shift
    elif direction == 'up':
        start_x, end_x, start_y, end_y = max_shift + shift, size + max_shift + shift, max_shift, size + max_shift
    else:  # direction == 'down'
        start_x, end_x, start_y, end_y = max_shift - shift, size + max_shift - shift, max_shift, size + max_shift

    # 平移图像
    x = padded_tensor[:, :, start_x:end_x, start_y:end_y]
    return x

def rand_zoom_with_mirror(x, ratio_range=(0.8, 1.2)):
    assert len(x.shape) == 4, 'Input tensor must be 4D'

    batchsize, channels, height, width = x.shape
    # 随机选择放大或缩小，并生成随机缩放比例
    zoom_ratio = random.uniform(*ratio_range)
    new_height = int(round(height * zoom_ratio))
    new_width = int(round(width * zoom_ratio))

    # 应用缩放
    zoomed_tensor = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # 调整尺寸以匹配原始尺寸
    if new_height != height or new_width != width:
        # 计算填充或裁剪的尺寸
        pad_height = (height - new_height) if new_height < height else 0
        pad_width = (width - new_width) if new_width < width else 0
        crop_height = (new_height - height) if new_height > height else 0
        crop_width = (new_width - width) if new_width > width else 0

        # 应用镜像填充或裁剪
        if pad_height > 0 or pad_width > 0:
            zoomed_tensor = F.pad(zoomed_tensor, (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2), mode='reflect')
        if crop_height > 0 or crop_width > 0:
            zoomed_tensor = zoomed_tensor[:, :, crop_height // 2: -crop_height // 2, crop_width // 2: -crop_width // 2]

    return zoomed_tensor

def rand_cutout(x, ratio=0.2):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij'

    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_zoom_translate_with_mirror(x, zoom_ratio_range=(0.8, 1.2), translation_ratio_range=(0, 0.2)):
    assert len(x.shape) == 4, 'Input tensor must be 4D'

    # 放大或缩小图像
    batchsize, channels, height, width = x.shape
    zoom_ratio = random.uniform(*zoom_ratio_range)
    # print(zoom_ratio)
    new_height = int(round(height * zoom_ratio))
    new_width = int(round(width * zoom_ratio))

    zoomed_tensor = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # 平移图像
    translation_ratio = random.uniform(*translation_ratio_range)
    # print(translation_ratio)
    max_shift = int(zoomed_tensor.shape[2] * translation_ratio + 0.5)
    direction = random.choice(['left', 'right', 'up', 'down'])
    shift = torch.randint(0, max_shift + 1, (1,)).item()
    # print(direction)
    # 为平移操作添加填充
    padded_tensor = F.pad(zoomed_tensor, (max_shift, max_shift, max_shift, max_shift), mode='reflect')

    if direction == 'left':
        start_x, end_x, start_y, end_y = max_shift, new_height + max_shift, max_shift + shift, new_width + max_shift + shift
    elif direction == 'right':
        start_x, end_x, start_y, end_y = max_shift, new_height + max_shift, max_shift - shift, new_width + max_shift - shift
    elif direction == 'up':
        start_x, end_x, start_y, end_y = max_shift + shift, new_height + max_shift + shift, max_shift, new_width + max_shift
    else:  # direction == 'down'
        start_x, end_x, start_y, end_y = max_shift - shift, new_height + max_shift - shift, max_shift, new_width + max_shift

    translated_tensor = padded_tensor[:, :, start_x:end_x, start_y:end_y]

    # 调整尺寸以匹配原始尺寸
    final_tensor = F.interpolate(translated_tensor, size=(height, width), mode='bilinear', align_corners=False)

    return final_tensor

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation_with_mirror],
    'zoom': [rand_zoom_with_mirror],
    'cutout': [rand_cutout],
    'zt': [rand_zoom_translate_with_mirror]
}

if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

        # 加载图像
    image_path = 'test.png'
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    image = transforms.ToPILImage()(image_tensor.squeeze(0))
    # 进行平移和镜像填充，使用 DiffAugment 函数
    # policy = 'translation,zoom'  # 设定 DiffAugment 的策略
    #paddedtrans_img = DiffAugment(image_tensor, policy)  # 使用 DiffAugment 处理图像
    da_img = rand_zoom_translate_with_mirror(image_tensor)

    # 将张量转换回图像
    da_img = transforms.ToPILImage()(da_img.squeeze(0))


    # # 显示原始图像和处理后的图像
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(image)
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.title("Translated Image with padding")
    plt.imshow(da_img)
    plt.axis('off')



    plt.show()
