import torch
import exr
import transformer.model as model
import argparse
import os
import sys
import utils
import math


# Implementation of Neural Graphics Texture Compression Supporting Random Access: https://arxiv.org/abs/2407.00021
def train_network(image: torch.Tensor):
    assert image.ndim == 3
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

    network = model.CompressionNetwork(channels)
    batch_tensor = image.new_empty(
        (batch_size, channels, stage_zero_crop_dim, stage_zero_crop_dim)
    )

    for training_step in range(total_steps):
        stage = (
            0
            if training_step < stage_zero_steps
            else (1 if training_step < stage_zero_steps + stage_one_steps else 2)
        )
        crop_dim = 256 if stage == 0 else 512
        max_mip = int.bit_length(crop_dim) - 1

        if training_step == stage_zero_steps:
            batch_tensor = image.new_empty(
                (batch_size, channels, max_crop_dim, max_crop_dim)
            )

        utils.random_crops_into(batch_tensor, image, crop_dim)

        u = torch.rand(()).item()
        mip = 0
        if u < 0.1:
            mip = int(torch.randint(0, max_mip + 1, ()).item())
        else:
            u = torch.rand(()).item()
            mip = math.floor(-math.log2(u) / 2)

        assert mip >= 0 and mip <= max_mip

        # See section 4 of paper
        encoded_tensor = network.global_transformation(batch_tensor)
        assert encoded_tensor.shape == (
            batch_size,
            network.channels_m,
            crop_dim // 8,
            crop_dim // 8,
        ), f"Unexpected encoded_tensor shape: {encoded_tensor.shape}"

        g0, g1 = network.grid_constructor_step(encoded_tensor)

        y0, y1, coords = network.grid_sample_step(g0, g1, crop_dim, mip)
        output = network.texture_synthesis_step(y0, y1, coords, crop_dim, mip)




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
