import torch
import torch.nn as nn

# Based on https://towardsdatascience.com/building-a-convolutional-neural-network-from-scratch-using-numpy-a22808a00a40


def _distinct_patch_generator(image_batch, kernel_size):
    batch_size, n_channels, image_h, image_w = image_batch.shape
    output_h = image_h // kernel_size
    output_w = image_w // kernel_size
    for h in range(output_h):
        for w in range(output_w):
            start_h = h * kernel_size
            end_h = start_h + kernel_size
            start_w = w * kernel_size
            end_w = start_w + kernel_size
            patch = image_batch[:, :, start_h:end_h, start_w:end_w]
            yield patch, h, w


class MaxPooling(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        batch_size, n_channels, image_h, image_w = x.shape
        output_h = image_h // self.kernel_size
        output_w = image_w // self.kernel_size
        output = torch.zeros(
            [batch_size, n_channels, output_h, output_w], device=x.device
        )
        for patch, h, w in _distinct_patch_generator(x, self.kernel_size):
            output[:, :, h, w] = torch.amax(patch, dim=(2, 3))
        return output
