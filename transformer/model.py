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
        num_frequencies=3,
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
        self.downsample = nn.AvgPool2d(kernel_size=2)

    def asymmetric_scalar_quantization(self, x, bits: 4):
        levels = 2**bits
        step = 1 / levels
        q_min = -(levels // 2) + 1
        q_max = levels // 2

        q = torch.round(x / step)
        q = torch.clamp(q, q_min, q_max)
        x_quantized = q * step

        assert x_quantized.shape == x.shape, (
            "Mismatched shapes after asymmetric quantization"
        )

        return x + (x_quantized - x).detach()

    def additive_uniform_noise_quantization(self, x, bits=4):
        half_width = 1 / (2 ** (bits + 1))
        noise = torch.empty_like(x).uniform_(-half_width, half_width)
        return x + noise

    def grid_constructor_step(
        self, x: torch.Tensor, stage: int, bits=4
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g0 = self.linear_projection_g0(x)
        g1 = self.linear_projection_g1(x)
        assert g0.shape[1] == self.grid_channels and g1.shape[1] == self.grid_channels

        if stage < 2:
            g0 = self.additive_uniform_noise_quantization(g0, bits)
            g1 = self.additive_uniform_noise_quantization(g1, bits)
        else:
            g0 = self.asymmetric_scalar_quantization(g0, bits)
            g1 = self.asymmetric_scalar_quantization(g1, bits)
        return g0, g1

    def gather_grid_corners(
        self, grid: torch.Tensor, corner_indices: torch.Tensor
    ) -> torch.Tensor:
        # grid: [B, C, grid_h, grid_w]
        # corner_indices: [4, output_h, output_w]
        batch_size, channels, _, _ = grid.shape
        num_corners, output_h, output_w = corner_indices.shape

        # grid_flat: [B, C, grid_h * grid_w]
        grid_flat = grid.flatten(2)

        # gather_indices: [B, C, 4 * output_h * output_w]
        gather_indices = corner_indices.reshape(1, 1, -1).expand(
            batch_size, channels, -1
        )

        # corners: [B, C, 4 * output_h * output_w] -> [B, C, 4, output_h, output_w]
        corners = torch.gather(grid_flat, 2, gather_indices)
        return corners.view(batch_size, channels, num_corners, output_h, output_w)

    def grid_sample_step(
        self, g0: torch.Tensor, g1: torch.Tensor, crop_dim, mip: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert g0.ndim == 4, "Sampled grid tensor should be [B, N, H, W]"
        assert g0.shape == g1.shape

        grid_w, grid_h = g0.shape[3], g0.shape[2]
        stride = 2 ** (max(mip, 3) - 3)
        output_w, output_h = crop_dim >> mip, crop_dim >> mip

        x = torch.arange(output_w, device=g0.device, dtype=torch.float32)
        y = torch.arange(output_h, device=g0.device, dtype=torch.float32)

        # TODO: I'm not sure if -0.5 does anything, since we're wrapping
        # Y0: Concat 4 corners from g0
        grid_pixel_x = (x + 0.5) / output_w * grid_w - 0.5
        grid_pixel_y = (y + 0.5) / output_h * grid_h - 0.5

        # NOTE: Wording in 4.3 is ambiguous. It suggests not snapping the top left corner
        # to stride, but then how are the interpolation weights calculated?
        grid_pixel_base_x = torch.floor(grid_pixel_x / stride) * stride
        grid_pixel_base_y = torch.floor(grid_pixel_y / stride) * stride
        #grid_pixel_base_x = torch.floor(grid_pixel_x)
        #grid_pixel_base_y = torch.floor(grid_pixel_y)

        pixel_x0 = grid_pixel_base_x.long() % grid_w
        pixel_x1 = (grid_pixel_base_x.long() + stride) % grid_w
        pixel_y0 = grid_pixel_base_y.long() % grid_h
        pixel_y1 = (grid_pixel_base_y.long() + stride) % grid_h

        # pixel_x*: [output_w], pixel_y*: [output_h]
        # Each expression broadcasts to [output_h, output_w] and stores y * grid_w + x.
        # corner_indices: [4, output_h, output_w], with order 00, 10, 01, 11.
        corner_indices = torch.stack(
            (
                pixel_y0[:, None] * grid_w + pixel_x0[None, :],
                pixel_y0[:, None] * grid_w + pixel_x1[None, :],
                pixel_y1[:, None] * grid_w + pixel_x0[None, :],
                pixel_y1[:, None] * grid_w + pixel_x1[None, :],
            ),
            dim=0,
        )

        # y0_corners: [B, C, 4, output_h, output_w]
        y0_corners = self.gather_grid_corners(g0, corner_indices)

        # y0: [B, 4 * C, output_h, output_w], corner-concatenated as 00, 10, 01, 11.
        y0 = y0_corners.permute(0, 2, 1, 3, 4).flatten(1, 2)
        assert y0.shape[1:] == (4 * self.grid_channels, output_h, output_w)

        # Y1: Bilerp 4 corners from g1
        # y1_corners: [B, C, 4, output_h, output_w]
        y1_corners = self.gather_grid_corners(g1, corner_indices)

        w1x = (grid_pixel_x - grid_pixel_base_x) / stride
        w1y = (grid_pixel_y - grid_pixel_base_y) / stride
        #w1x = (grid_pixel_x - grid_pixel_base_x)
        #w1y = (grid_pixel_y - grid_pixel_base_y)
        w0x = 1.0 - w1x
        w0y = 1.0 - w1y

        w1x = w1x.view(1, 1, 1, output_w)
        w0x = w0x.view(1, 1, 1, output_w)

        w1y = w1y.view(1, 1, output_h, 1)
        w0y = w0y.view(1, 1, output_h, 1)

        # corner_weights: [1, 1, 4, output_h, output_w], matching 00, 10, 01, 11.
        corner_weights = torch.stack(
            (w0x * w0y, w1x * w0y, w0x * w1y, w1x * w1y), dim=2
        )

        # y1: [B, C, output_h, output_w]
        y1 = (y1_corners * corner_weights).sum(dim=2)
        assert y1.shape[1:] == (self.grid_channels, output_h, output_w)

        pe_scale_x = 0.5 * grid_w / output_w
        pe_scale_y = 0.5 * grid_h / output_h
        pe_x = (x + 0.5) * pe_scale_x
        pe_y = (y + 0.5) * pe_scale_y

        pe_yy, pe_xx = torch.meshgrid(pe_y, pe_x, indexing="ij")
        coords = torch.stack((pe_xx, pe_yy), dim=-1)
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
        wave_coords = coords
        encoding_waves = []
        for _ in range(self.num_frequencies):
            encoding_waves.append(torch.frac(wave_coords) * 2.0 - 1.0)
            encoding_waves.append(torch.frac(wave_coords + 0.25) * 2.0 - 1.0)
            wave_coords = wave_coords * 2.0

        positional_encoding = torch.cat(encoding_waves, dim=-1)
        positional_encoding = positional_encoding.permute(
            0, 3, 1, 2
        ).to(y0.dtype)  # [B, 4 * F, H, W]

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

    def forward(self, batch_tensor: torch.Tensor, mip, stage: int) -> torch.Tensor:
        assert batch_tensor.shape[2] == batch_tensor.shape[3]
        crop_dim = batch_tensor.shape[2]
        encoded = self.global_transformation(batch_tensor)
        assert encoded.shape == (
            batch_tensor.shape[0],
            self.channels_m,
            crop_dim // 8,
            crop_dim // 8,
        ), f"Unexpected encoded_tensor shape: {encoded.shape}"
        g0, g1 = self.grid_constructor_step(encoded, stage)
        y0, y1, coords = self.grid_sample_step(g0, g1, crop_dim, mip)
        return self.texture_synthesis_step(y0, y1, coords, crop_dim, mip)
