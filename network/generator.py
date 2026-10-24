import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .discriminator import *
import random

class SpatialVariationModule(nn.Module):
    def __init__(self):
        super(SpatialVariationModule, self).__init__()

    def corner_center_segmentation(self, spa):
        batch_size, channels, height, width = spa.shape

        center_y, center_x = height // 2, width // 2
        center_size = 3
        cy_start, cy_end = max(0, center_y - center_size // 2), min(height, center_y + center_size // 2)
        cx_start, cx_end = max(0, center_x - center_size // 2), min(width, center_x + center_size // 2)
        center_region = spa[:, :, cy_start:cy_end, cx_start:cx_end]
        center_features = torch.mean(center_region.reshape(batch_size, channels, -1), dim=-1)  # (256, 1)

        edge_width = 1
        top_edge = spa[:, :, :edge_width, :]
        bottom_edge = spa[:, :, -edge_width:, :]
        left_edge = spa[:, :, edge_width:-edge_width, :edge_width]
        right_edge = spa[:, :, edge_width:-edge_width, -edge_width:]

        edge_features = torch.cat([
            top_edge.reshape(batch_size, channels, -1),
            bottom_edge.reshape(batch_size, channels, -1),
            left_edge.reshape(batch_size, channels, -1),
            right_edge.reshape(batch_size, channels, -1)], dim=-1)
        edge_mean = torch.mean(edge_features, dim=-1)  # (256, 1)

        center_plane = center_features.unsqueeze(-1).unsqueeze(-1)  # (256, 1, 1, 1)
        center_plane = center_plane.expand(batch_size, channels, height, width)

        edge_plane = edge_mean.unsqueeze(-1).unsqueeze(-1)  # (256, 1, 1, 1)
        edge_plane = edge_plane.expand(batch_size, channels, height, width)

        center_similarity = torch.abs(spa - center_plane)
        edge_similarity = torch.abs(spa - edge_plane)

        diff_score = center_similarity - edge_similarity
        center_masks = (diff_score < 0).float()
        background_masks = 1.0 - center_masks

        return center_masks, background_masks

    def compute_mixing_weights(self, spa, center_masks, background_masks):
        center_images = spa * center_masks
        background_images = spa * background_masks

        def compute_nonzero_mean(images):
            flattened = images.view(images.size(0), images.size(1), -1)
            nonzero_mask = (flattened != 0)
            sum_nonzero = torch.where(nonzero_mask, flattened, torch.zeros_like(flattened)).sum(dim=2, keepdim=True)
            count_nonzero = nonzero_mask.sum(dim=2, keepdim=True).float()
            count_nonzero = torch.where(count_nonzero == 0, torch.ones_like(count_nonzero), count_nonzero)
            return sum_nonzero / count_nonzero

        center_mean = compute_nonzero_mean(center_images)
        background_mean = compute_nonzero_mean(background_images)
        diff = torch.abs(center_mean - background_mean).unsqueeze(-1)

        return center_images, background_images, diff

    def index_swap(self, center_images, background_images, diff):
        batch_size = center_images.size(0)

        diff_values = diff.view(batch_size)
        sorted_indices = torch.argsort(diff_values, descending=True)

        num_swap = int(batch_size * 0.4)
        swap_indices = sorted_indices[:num_swap]

        new_center_images = center_images.clone()
        new_background_images = background_images.clone()

        if num_swap > 1:
            perm_center = swap_indices[torch.randperm(len(swap_indices))]
            perm_background = swap_indices[torch.randperm(len(swap_indices))]

            new_center_images[swap_indices] = center_images[perm_center]
            new_background_images[swap_indices] = background_images[perm_background]

        new_images = new_center_images + new_background_images

        return new_images

    def forward(self, spa):
        if self.training:
            center_masks, background_masks = self.corner_center_segmentation(spa)
            center_images, background_images, diff = self.compute_mixing_weights(spa, center_masks, background_masks)
            new_images = self.index_swap(center_images, background_images, diff)
            return new_images
        else:
            return spa


class SpeRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.register_buffer('channels_64', torch.arange(64))
        self.register_buffer('channels_32', torch.arange(32))
        self.register_buffer('channels_16', torch.arange(16))
        self.register_buffer('channels_8', torch.arange(8))

        self.num_scales = 4
        self.attention_convs = nn.ModuleList([nn.Conv2d(64, 64, kernel_size=1) for _ in range(self.num_scales)])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=1)

    def randomize_channels(self):
        perm = torch.randperm(64, device=self.channels_64.device)

        self.channels_64 = torch.sort(perm).values
        self.channels_32 = torch.sort(perm[torch.randperm(64)[:32]]).values
        self.channels_16 = torch.sort(perm[torch.randperm(64)[:16]]).values
        self.channels_8 = torch.sort(perm[torch.randperm(64)[:8]]).values

    def forward(self, x):
        if self.training:
            i = 0
            i = i + 1
            if i % 10 == 0:
                self.randomize_channels()
                i = 0
            x_64 = x[:, self.channels_64]  # [N, 64, H, W]
            x_32 = x[:, self.channels_32]  # [N, 32, H, W]
            x_16 = x[:, self.channels_16]  # [N, 16, H, W]
            x_8 = x[:, self.channels_8]  # [N, 8, H, W]

            def spectral_randomize(features):
                N, C, H, W = features.size()
                idx_swap = torch.randperm(N, device=x.device)
                features_flat = features.view(N, C, -1)

                mean = features_flat.mean(1, keepdim=True)  # [N, C, 1]
                var = features_flat.var(1, keepdim=True)  # [N, C, 1]

                features_norm = (features_flat - mean) / (var + self.eps).sqrt()

                features_swapped = features_norm[idx_swap].detach()
                features_swapped = features_swapped * (var + self.eps).sqrt() + mean

                return features_swapped.view(N, C, H, W)

            x_64_processed = spectral_randomize(x_64)
            x_32_processed = spectral_randomize(x_32)
            x_16_processed = spectral_randomize(x_16)
            x_8_processed = spectral_randomize(x_8)

            features_list = []
            x_64_aligned = x_64_processed
            features_list.append(x_64_aligned)

            for channels, processed in [(self.channels_32, x_32_processed),(self.channels_16, x_16_processed),(self.channels_8, x_8_processed)]:
                aligned = torch.zeros_like(x_64_processed)
                aligned[:, channels] = processed
                features_list.append(aligned)

            attention_maps = []
            for i, (features, conv) in enumerate(zip(features_list, self.attention_convs)):
                attention_map = conv(features)  # [N, 64, H, W]
                attention_maps.append(attention_map)

            attention_stack = torch.stack(attention_maps, dim=1)  # [N, 4, 64, H, W]
            #attention_weights = self.softmax(attention_stack)  # [N, 4, 64, H, W]

            features_stack = torch.stack(features_list, dim=1)  # [N, 4, 64, H, W]
            features_stack = self.softmax(features_stack)
            x_sum = (features_stack * attention_stack).sum(dim=1)  # [N, 64, H, W]

            x_sum = x_sum + self.fusion_conv(x_sum)

            return x_sum
        else:
            return x


class Generator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=10, device=0):
        super().__init__()
        stride = (kernelsize - 1) // 2
        self.zdim = zdim
        self.imdim = imdim
        self.imsize = imsize
        self.device = device


        self.conv_spa1 = nn.Conv2d(imdim, 1, 1, 1)
        self.conv_spa2 = nn.Conv2d(1, n, 1, 1)

        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0])

        self.conv1 = nn.Conv2d(n + n , n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, imdim, kernelsize, 1, stride)

        self.speRandom = SpeRandomization(n)
        self.SpatialVariationModule = SpatialVariationModule()

    def forward(self, x):
        x_spa = self.conv_spa1(x)
        x_spe = F.relu(self.conv_spe1(x))

        x_spa= self.SpatialVariationModule(x_spa)
        x_spe1 = self.speRandom(x_spe)

        x_spe = self.conv_spe2(x_spe1)
        x_spa = self.conv_spa2(x_spa)
        x = F.relu(self.conv1(torch.cat((x_spa, x_spe), 1)))
        x = torch.sigmoid(self.conv2(x))

        return x