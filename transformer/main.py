import torch
import torch.nn as nn
import exr
import argparse
import os
import sys


class HalfTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.tanh(x)


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # TODO: 2x or 4x?
        midChannels = min(channels, channels) // 2

        self.layer0 = nn.Conv2d(
            in_channels=channels,
            out_channels=midChannels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.layer1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(
            in_channels=midChannels,
            out_channels=midChannels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer3 = nn.ReLU(inplace=True)
        self.layer4 = nn.Conv2d(
            in_channels=midChannels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out + x


class LinearResidualBlock(nn.Module):
    def __init__(self, input_channels, lrb_channels=32):
        super().__init__()
        self.layer0 = nn.Conv2d(
            in_channels=input_channels, out_channels=lrb_channels, kernel_size=1
        )
        self.layer1 = nn.LeakyReLU(inplace=True)
        self.layer2 = nn.Conv2d(
            in_channels=lrb_channels, out_channels=lrb_channels, kernel_size=1
        )
        self.layer3 = nn.LeakyReLU(inplace=True)
        self.layer4 = nn.Conv2d(
            in_channels=lrb_channels, out_channels=input_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out + x


class CompressionNetwork(nn.Module):
    def __init__(
        self,
        input_channels,
        channels_n=192,
        channels_m=320,
        grid_channels=32,
        num_frequencies=6,
    ):
        super().__init__()
        self.grid_channels = grid_channels
        self.channels_m = channels_m
        self.num_frequencies = num_frequencies
        self.global_transformation = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=channels_n,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            ResidualBottleneckBlock(channels_n),
            ResidualBottleneckBlock(channels_n),
            ResidualBottleneckBlock(channels_n),
            nn.Conv2d(
                in_channels=channels_n,
                out_channels=channels_n,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            ResidualBottleneckBlock(channels_n),
            ResidualBottleneckBlock(channels_n),
            ResidualBottleneckBlock(channels_n),
            nn.Conv2d(
                in_channels=channels_n,
                out_channels=channels_m,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            HalfTanh(),
        )

        self.linear_projection_g0 = nn.Conv2d(
            in_channels=channels_m,
            out_channels=grid_channels,
            kernel_size=1,
            stride=1,
        )

        self.linear_projection_g1 = nn.Conv2d(
            in_channels=channels_m,
            out_channels=grid_channels,
            kernel_size=1,
            stride=1,
        )

        texture_synthesis_num_input_features = (
            5 * grid_channels + num_frequencies * 4 + 1
        )
        linear_residual_block_num_channels = 32
        self.texture_synthesizer = nn.Sequential(
            nn.Conv2d(
                in_channels=texture_synthesis_num_input_features,
                out_channels=linear_residual_block_num_channels,
                kernel_size=1,
            ),
            LinearResidualBlock(linear_residual_block_num_channels),
            LinearResidualBlock(linear_residual_block_num_channels),
            LinearResidualBlock(linear_residual_block_num_channels),
            LinearResidualBlock(linear_residual_block_num_channels),
            nn.Conv2d(
                in_channels=linear_residual_block_num_channels,
                out_channels=input_channels,
                kernel_size=1,
            ),
        )

    def asymmetric_scalar_quantization(self, x, bits: 4):
        q_min = -((2**bits) - 1) / (2 ** (bits + 1))
        q_max = 0.5
        levels = 2**bits
        scale = (q_max - q_min) / (levels - 1)

        x_clamped = torch.clamp(x, q_min, q_max)
        x_quantized = torch.round((x_clamped - q_min) / scale) * scale + q_min

        assert x_quantized.shape == x.shape, (
            "Mismatched shapes after asymmetric quantization"
        )

        return x + (x_quantized - x).detach()

    def grid_constructor_step(
        self, x: torch.Tensor, bits: 4
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g0 = self.linear_projection_g0(x)
        g1 = self.linear_projection_g1(x)
        assert g0.shape[1] == self.grid_channels and g1.shape[1] == self.grid_channels

        g0 = self.asymmetric_scalar_quantization(g0, bits)
        g1 = self.asymmetric_scalar_quantization(g1, bits)
        return g0, g1

    def grid_sample_step(
        self, g0: torch.Tensor, g1: torch.Tensor, crop_dim, mip
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert g0.ndim == 4, "Sampled grid tensor should be [B, N, H, W]"
        assert g0.shape == g1.shape

        grid_w, grid_h = g0.shape[3], g0.shape[2]
        stride = 2 ** (max(mip, 3) - 3)

        x = torch.arange(crop_dim, device=g0.device, dtype=g0.dtype)
        y = torch.arange(crop_dim, device=g0.device, dtype=g0.dtype)

        # TODO: I'm not sure if -0.5 does anything, since we're wrapping
        # Y0: Concat 4 corners from g0
        grid_pixel_x = (x + 0.5) / crop_dim * grid_w - 0.5
        grid_pixel_y = (y + 0.5) / crop_dim * grid_h - 0.5

        grid_pixel_floor_x = torch.floor(grid_pixel_x)
        grid_pixel_floor_y = torch.floor(grid_pixel_y)

        grid_base_uv_x = ((grid_pixel_floor_x + 0.5) / grid_w) * 2.0 - 1.0
        grid_base_uv_y = ((grid_pixel_floor_y + 0.5) / grid_h) * 2.0 - 1.0

        yy, xx = torch.meshgrid(grid_base_uv_y, grid_base_uv_x, indexing="ij")
        corner00 = torch.stack((xx, yy), dim=-1)
        corner00 = corner00.unsqueeze(0).expand(g0.shape[0], -1, -1, -1)

        dx = 2.0 * stride / grid_w
        dy = 2.0 * stride / grid_h

        corner10 = corner00 + g0.new_tensor([dx, 0])
        corner01 = corner00 + g0.new_tensor([0, dy])
        corner11 = corner00 + g0.new_tensor([dx, dy])

        y0_corner00 = nn.functional.grid_sample(
            input=g0,
            grid=corner00,
            mode="nearest",
            padding_mode="reflection",
            align_corners=False,
        )

        y0_corner10 = nn.functional.grid_sample(
            input=g0,
            grid=corner10,
            mode="nearest",
            padding_mode="reflection",
            align_corners=False,
        )

        y0_corner01 = nn.functional.grid_sample(
            input=g0,
            grid=corner01,
            mode="nearest",
            padding_mode="reflection",
            align_corners=False,
        )

        y0_corner11 = nn.functional.grid_sample(
            input=g0,
            grid=corner11,
            mode="nearest",
            padding_mode="reflection",
            align_corners=False,
        )

        y0 = torch.cat((y0_corner00, y0_corner10, y0_corner01, y0_corner11), dim=1)
        assert y0.shape[1:] == [4 * self.grid_channels, crop_dim, crop_dim]

        # Y1: Bilerp 4 corners from g1
        y1_corner00 = nn.functional.grid_sample(
            input=g1,
            grid=corner00,
            mode="nearest",
            padding_mode="reflection",
            align_corners=False,
        )

        y1_corner10 = nn.functional.grid_sample(
            input=g1,
            grid=corner10,
            mode="nearest",
            padding_mode="reflection",
            align_corners=False,
        )

        y1_corner01 = nn.functional.grid_sample(
            input=g1,
            grid=corner01,
            mode="nearest",
            padding_mode="reflection",
            align_corners=False,
        )

        y1_corner11 = nn.functional.grid_sample(
            input=g1,
            grid=corner11,
            mode="nearest",
            padding_mode="reflection",
            align_corners=False,
        )

        grid_uv_x = (x + 0.5) / crop_dim
        grid_uv_y = (y + 0.5) / crop_dim

        # TODO: are these the right weights?
        w1x = grid_uv_x - grid_pixel_floor_x
        w1y = grid_uv_y - grid_pixel_floor_y
        w0x = 1.0 - w1x
        w0y = 1.0 - w1y

        w1x = w1x.view(1, 1, 1, crop_dim)
        w0x = w0x.view(1, 1, 1, crop_dim)

        w1y = w1y.view(1, 1, crop_dim, 1)
        w0y = w0y.view(1, 1, crop_dim, 1)

        y1 = (
            y1_corner00 * w0x * w0y
            + y1_corner10 * w1x * w0y
            + y1_corner01 * w0x * w1y
            + y1_corner11 * w1x * w1y
        )
        assert y1.shape[1:] == [self.grid_channels, crop_dim, crop_dim]

        return y0, y1

    def texture_synthesis_step(
        self, y0: torch.Tensor, y1: torch.Tensor, coords, crop_dim, mip
    ):
        # y0: [B, 4 * cg0, H, W]
        # y1: [B, cg1, H, W]
        # coords: [B, H, W, 2]

        assert y0.shape[0] == y1.shape[1] and y0.shape[2:] == y1.shape[2:] == (
            crop_dim,
            crop_dim,
        )
        batch_size = y0.shape[0]
        max_mip = int.bit_length(crop_dim) - 1
        pow2s = 2.0 ** torch.arange(
            self.num_frequencies, device=y0.device, dtype=y0.dtype
        )  # [F], F=num_frequencies

        vals = (
            pow2s[:, None] * coords[..., None, :] * torch.pi
        )  # [F, 1] * [B, H, W, 1, 2] = [B, H, W, F, 2]

        positional_encoding = torch.cat(
            (torch.sin(vals), torch.cos(vals)), dim=-1
        )  # [B, H, W, F, 4]
        positional_encoding = positional_encoding.flatten(-2)  # [B, H, W, 4 * F]
        positional_encoding = positional_encoding.permute(
            0, 3, 1, 2
        )  # [B, 4 * F, H, W]

        mip_tensor = y0.new_full((batch_size, 1, crop_dim, crop_dim), mip / max_mip)

        d_theta = torch.cat(
            [y0, y1, mip_tensor, positional_encoding], dim=1
        )  # [B, 4 * cg0 + cg1 + 1 + 4 * F, H, W]

        assert d_theta.shape == [
            batch_size,
            5 * self.grid_channels + 1 + 4 * self.num_frequencies,
            crop_dim,
            crop_dim,
        ]

        d = self.texture_synthesizer(d_theta)
        return d


# Implementation of Neural Graphics Texture Compression Supporting Random Access: https://arxiv.org/abs/2407.00021
def train_network(image: torch.Tensor):
    channels, height, width = image.shape
    stage_zero_crop_dim = 256
    max_crop_dim = 512
    batch_size = 4
    stage_zero_steps = 160000
    stage_one_steps = 80000
    stage_two_steps = 20000
    total_steps = stage_zero_steps + stage_one_steps + stage_two_steps

    assert height >= max_crop_dim and width >= max_crop_dim, (
        f"Width/height of image is less than min supported size (512): {width}, {height}"
    )

    network = CompressionNetwork(channels)
    batch_tensor = image.new_empty(
        (batch_size, channels, stage_zero_steps, stage_zero_crop_dim)
    )

    for training_step in range(total_steps):
        stage = (
            0
            if training_step < stage_zero_steps
            else (1 if training_step < stage_zero_steps + stage_one_steps else 2)
        )
        crop_dim = 256 if stage == 0 else 512

        if training_step == stage_zero_steps:
            batch_tensor = image.new_empty(
                (batch_size, channels, max_crop_dim, max_crop_dim)
            )
        for batch_index in range(batch_size):
            start_u = torch.randint(0, width - crop_dim + 1)
            start_v = torch.randint(0, height - crop_dim + 1)
            batch_tensor[batch_index].copy_(
                image[:, start_v : start_v + crop_dim, start_u : start_u + crop_dim]
            )
        # See section 4 of paper
        encoded_tensor = network.global_transformation(batch_tensor)
        assert encoded_tensor.shape == [
            batch_size,
            network.channels_m,
            crop_dim / 8,
            crop_dim / 8,
        ], f"Unexpected encoded_tensor shape: {encoded_tensor.shape}"

        g0, g1 = network.grid_constructor_step(encoded_tensor)

        u = torch.linspace(
            start=-1.0, end=1.0, steps=crop_dim, device=image.device, dtype=image.dtype
        )
        v = torch.linspace(
            start=-1.0, end=1.0, steps=crop_dim, device=image.device, dtype=image.dtype
        )

        vv, uu = torch.meshgrid(v, u, indexing="ij")
        grid_indices = torch.stack((uu, vv), dim=-1)  # (H, W, 2)
        assert grid_indices.shape == [crop_dim, crop_dim, 2]
        # TODO: same values for every item in batch??
        grid_indices = grid_indices.unsqueeze(0).expand(batch_size, -1, -1, -1)

        padding_mode = "reflection"
        align_corners = False

        # assert y0.shape == [batch_size, 4 * network.grid_channels, crop_dim * crop_dim], "Wrong Y0 shape"
        # y0 = y0.view(batch_size, 4 * network.grid_channels, crop_dim, crop_dim)

        # Texture synthesis (4.4)


def main():
    parser = argparse.ArgumentParser(
        description="Implementation of Neural Graphics Texture Compression Supporting Random Access"
    )
    parser.add_argument("filename", help="The name of the file to compress")

    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        print(f"Error: {args.filename} is not a valid file")
        sys.exit()

    filename, file_extension = os.path.splitext(args.filename)
    tensor: torch.Tensor = None
    if file_extension == ".exr":
        print(f"Valid file: {args.filename}")
        tensor = exr.ParseEXR(args.filename)
        print(f"{tensor.shape}")
    else:
        print(f"${file_extension} files are currently not supported.")
        sys.exit()

    torch.manual_seed(1337)


if __name__ == "__main__":
    main()
