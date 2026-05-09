import torch
import torch.nn as nn


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
        self, g0: torch.Tensor, g1: torch.Tensor, crop_dim, mip: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert g0.ndim == 4, "Sampled grid tensor should be [B, N, H, W]"
        assert g0.shape == g1.shape

        grid_w, grid_h = g0.shape[3], g0.shape[2]
        stride = 2 ** (max(mip, 3) - 3)
        output_w, output_h = crop_dim >> mip, crop_dim >> mip

        x = torch.arange(output_w, device=g0.device, dtype=g0.dtype)
        y = torch.arange(output_h, device=g0.device, dtype=g0.dtype)

        # TODO: I'm not sure if -0.5 does anything, since we're wrapping
        # Y0: Concat 4 corners from g0
        grid_pixel_x = (x + 0.5) / output_w * grid_w - 0.5
        grid_pixel_y = (y + 0.5) / output_h * grid_h - 0.5

        # NOTE: Wording in 4.3 is ambiguous. It suggests not snapping the top left corner 
        # to stride, but then how are the interpolation weights calculated?
        grid_pixel_base_x = torch.floor(grid_pixel_x / stride) * stride
        grid_pixel_base_y = torch.floor(grid_pixel_y / stride) * stride

        pixel_x0 = grid_pixel_base_x.long() % grid_w
        pixel_x1 = (grid_pixel_base_x.long() + stride) % grid_w
        pixel_y0 = grid_pixel_base_y.long() % grid_h
        pixel_y1 = (grid_pixel_base_y.long() + stride) % grid_h

        y0_corner00 = g0[:, :, pixel_y0[:, None], pixel_x0[None, :]]
        y0_corner10 = g0[:, :, pixel_y0[:, None], pixel_x1[None, :]]
        y0_corner01 = g0[:, :, pixel_y1[:, None], pixel_x0[None, :]]
        y0_corner11 = g0[:, :, pixel_y1[:, None], pixel_x1[None, :]]

        y0 = torch.cat((y0_corner00, y0_corner10, y0_corner01, y0_corner11), dim=1)
        assert y0.shape[1:] == (4 * self.grid_channels, output_h, output_w)

        # Y1: Bilerp 4 corners from g1
        y1_corner00 = g1[:, :, pixel_y0[:, None], pixel_x0[None, :]]
        y1_corner10 = g1[:, :, pixel_y0[:, None], pixel_x1[None, :]]
        y1_corner01 = g1[:, :, pixel_y1[:, None], pixel_x0[None, :]]
        y1_corner11 = g1[:, :, pixel_y1[:, None], pixel_x1[None, :]]

        w1x = (grid_pixel_x - grid_pixel_base_x) / stride
        w1y = (grid_pixel_y - grid_pixel_base_y) / stride
        w0x = 1.0 - w1x
        w0y = 1.0 - w1y

        w1x = w1x.view(1, 1, 1, output_w)
        w0x = w0x.view(1, 1, 1, output_w)

        w1y = w1y.view(1, 1, output_h, 1)
        w0y = w0y.view(1, 1, output_h, 1)

        y1 = (
            y1_corner00 * w0x * w0y
            + y1_corner10 * w1x * w0y
            + y1_corner01 * w0x * w1y
            + y1_corner11 * w1x * w1y
        )
        assert y1.shape[1:] == (self.grid_channels, output_h, output_w)

        # TODO: maybe arbitrary
        grid_uv_x = (x + 0.5) / output_w
        grid_uv_y = (y + 0.5) / output_h
        grid_uv_x = grid_uv_x * 2.0 - 1.0
        grid_uv_y = grid_uv_y * 2.0 - 1.0

        uv_yy, uv_xx = torch.meshgrid(grid_uv_y, grid_uv_x, indexing="ij")
        coords = torch.stack((uv_xx, uv_yy), dim=-1)
        coords = coords.unsqueeze(0).expand(g0.shape[0], -1, -1, -1)

        return y0, y1, coords

    def texture_synthesis_step(
        self, y0: torch.Tensor, y1: torch.Tensor, coords, crop_dim, mip
    ):
        # y0: [B, 4 * cg0, H, W]
        # y1: [B, cg1, H, W]
        # coords: [B, H, W, 2]

        assert y0.shape[0] == y1.shape[0] and y0.shape[2:] == y1.shape[2:] == (
            crop_dim >> mip,
            crop_dim >> mip,
        )
        assert y0.shape[1] == 4 * self.grid_channels
        assert y1.shape[1] == self.grid_channels
        assert (
            coords.ndim == 4
            and coords.shape[0] == y0.shape[0]
            and coords.shape[1] == crop_dim >> mip
            and coords.shape[2] == crop_dim >> mip
            and coords.shape[3] == 2
        )

        batch_size = y0.shape[0]
        output_h, output_w = y0.shape[2], y0.shape[3]
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

        mip_tensor = y0.new_full((batch_size, 1, output_h, output_w), mip / max_mip)

        d_theta = torch.cat(
            [y0, y1, mip_tensor, positional_encoding], dim=1
        )  # [B, 4 * cg0 + cg1 + 1 + 4 * F, H, W]

        assert d_theta.shape == (
            batch_size,
            5 * self.grid_channels + 1 + 4 * self.num_frequencies,
            output_h,
            output_w,
        )

        d = self.texture_synthesizer(d_theta)
        return d
