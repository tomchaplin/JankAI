import torch
import torch.nn as nn

# Based on https://towardsdatascience.com/building-a-convolutional-neural-network-from-scratch-using-numpy-a22808a00a40


def _overlapping_patch_generator(image_batch, kernel_size):
    batch_size, image_h, image_w = image_batch.shape
    for h in range(image_h - kernel_size + 1):
        for w in range(image_w - kernel_size + 1):
            patch = image_batch[:, h : (h + kernel_size), w : (w + kernel_size)]
            yield patch, h, w


class Conv2d(nn.Module):
    def __init__(self, kernel_num, kernel_size):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        kernels = torch.Tensor(kernel_num, kernel_size, kernel_size)
        nn.init.normal_(kernels, mean=0.0, std=1.0 / (kernel_size**2))
        # Wrap in nn.Parameter to let PyTorch know it needs updating
        self.kernels = nn.Parameter(kernels)

    def forward(self, x):
        batch_size, image_h, image_w = x.shape
        conv_output = torch.zeros(
            [
                batch_size,
                self.kernel_num,
                image_h - self.kernel_size + 1,
                image_w - self.kernel_size + 1,
            ],
            device=x.device,
        )
        for patch, h, w in _overlapping_patch_generator(x, self.kernel_size):
            assert patch.shape == (batch_size, self.kernel_size, self.kernel_size)
            assert self.kernels.shape == (
                self.kernel_num,
                self.kernel_size,
                self.kernel_size,
            )
            patch = torch.unsqueeze(patch, dim=1)
            # patch.shape == (batch_size, 1, self.kernel_size, self.kernel_size)
            # self.kernel.shape == (self.kernel_num, self.kernel_size, self.kernel_size)
            # So multiplication broadcasts over batch_size and kernel_num
            mult = patch * self.kernels
            assert mult.shape == (
                batch_size,
                self.kernel_num,
                self.kernel_size,
                self.kernel_size,
            )
            convolved = mult.sum(dim=(2, 3))
            # Alternatively convolved = torch.einsum('bhw,khw->bk', patch, self.kernels)
            # convolved_(i, j) is a patch (with top right coords (h, w)) of image_i convolved with kernel_j
            assert convolved.shape == (batch_size, self.kernel_num)
            conv_output[:, :, h, w] = convolved
        return conv_output


# Following the recipe of https://pytorch.org/docs/stable/notes/extending.html
class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_batch, kernels):
        ctx.save_for_backward(image_batch, kernels)
        batch_size, image_h, image_w = image_batch.shape
        kernel_num, kernel_h, kernel_w = kernels.shape
        assert kernel_h == kernel_w
        conv_output = torch.zeros(
            [
                batch_size,
                kernel_num,
                image_h - kernel_h + 1,
                image_w - kernel_w + 1,
            ],
            device=image_batch.device,
        )
        for patch, h, w in _overlapping_patch_generator(image_batch, kernel_w):
            conv_output[:, :, h, w] = torch.einsum("bhw,khw->bk", patch, kernels)
        return conv_output

    @staticmethod
    def backward(ctx, grad_output):
        image_batch, kernels = ctx.saved_tensors
        grad_image_batch = grad_kernels = None

        kernel_num, kernel_h, kernel_w = kernels.shape

        def _compute_grad_image_batch():
            grad_image_batch = torch.zeros(image_batch.shape, device=image_batch.device)

            for patch, h, w in _overlapping_patch_generator(image_batch, kernel_w):
                grad_image_batch[
                    :, h : h + (kernel_h), w : (w + kernel_w)
                ] += torch.einsum("khw,bk->bhw", kernels, grad_output[:, :, h, w])

            return grad_image_batch

        def _compute_grad_kernels():
            grad_kernels = torch.zeros(kernels.shape, device=kernels.device)

            for patch, h, w in _overlapping_patch_generator(image_batch, kernel_w):
                grad_kernels += torch.einsum(
                    "bhw,bk->khw", patch, grad_output[:, :, h, w]
                )

            return grad_kernels

        if ctx.needs_input_grad[0]:
            grad_image_batch = _compute_grad_image_batch()
        if ctx.needs_input_grad[1]:
            grad_kernels = _compute_grad_kernels()

        return grad_image_batch, grad_kernels


class Conv2dFunctionWrapped(nn.Module):
    def __init__(self, kernel_num, kernel_size):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        kernels = torch.Tensor(kernel_num, kernel_size, kernel_size)
        nn.init.normal_(kernels, mean=0.0, std=1.0 / (kernel_size**2))
        # Wrap in nn.Parameter to let PyTorch know it needs updating
        self.kernels = nn.Parameter(kernels)

    def forward(self, x):
        return Conv2dFunction.apply(x, self.kernels)
