import torch
import exr
import model
import argparse
import os
import sys
import utils
import math
import time
from pathlib import Path
from PIL import Image
from torchvision import transforms


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

    network = model.CompressionNetwork(channels).to(image.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    use_amp = image.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch_tensor = image.new_empty(
        (batch_size, channels, stage_zero_crop_dim, stage_zero_crop_dim)
    )

    start_time = time.perf_counter()
    eval_max_mip = int.bit_length(max_crop_dim) - 1

    for training_step in range(total_steps):
        if training_step % 1000 == 0:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                preview_tensor = image.unsqueeze(0)
                preview_output = network(preview_tensor, 0, 2)
                target_mip = preview_tensor
                total_sse = image.new_zeros(())
                total_count = 0
                for eval_mip in range(eval_max_mip + 1):
                    mip_output = preview_output if eval_mip == 0 else network(preview_tensor, eval_mip, 0)
                    diff = (mip_output - target_mip).float()
                    total_sse = total_sse + torch.sum(diff * diff)
                    total_count += target_mip.numel()
                    if eval_mip < eval_max_mip:
                        target_mip = network.downsample(target_mip)
                total_mip_loss = total_sse / total_count
                psnr = -10.0 * torch.log10(total_mip_loss)
            preview_output = preview_output[0].permute(1, 2, 0).detach().cpu().numpy()
            output_path = Path(__file__).with_name(f"test{training_step}.exr")
            exr.pyexr.write(output_path, preview_output)
            print(
                f"Total mip loss: {total_mip_loss.item():.8f} "
                f"PSNR: {psnr.item():.2f} dB"
            )

        optimizer.zero_grad(set_to_none=True)
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
            optimizer.param_groups[0]["lr"] = 5e-5
        elif training_step == stage_zero_steps + stage_one_steps:
            optimizer.param_groups[0]["lr"] = 1e-5

        utils.random_crops_into(batch_tensor, image, crop_dim)

        u = torch.rand(()).item()
        mip = 0
        if u < 0.1:
            mip = int(torch.randint(0, max_mip + 1, ()).item())
        else:
            u = torch.rand(()).item()
            mip = min(math.floor(-math.log2(u) / 2), max_mip)

        assert mip >= 0 and mip <= max_mip

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = network(batch_tensor, mip, stage)

            target_tensor = batch_tensor
            for _ in range(mip):
                target_tensor = network.downsample(target_tensor)

            assert (
                target_tensor.shape
                == output.shape
                == (
                    batch_size,
                    channels,
                    crop_dim >> mip,
                    crop_dim >> mip,
                )
            )

            loss = torch.nn.functional.mse_loss(output, target_tensor)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if training_step % 1000 == 0:
            t = time.perf_counter()
            delta = t - start_time
            start_time = t
            print(f"Step: {training_step} Time: {delta:.2f}s")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        output = network(image.unsqueeze(0), 0, 2)
    output = output[0].permute(1, 2, 0).detach().cpu().numpy()

    output_path = Path(__file__).with_name("test.exr")
    exr.pyexr.write(output_path, output)


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
    elif file_extension == ".jpg":
        img = Image.open(args.filename).convert("RGB")
        tensor = transforms.ToTensor()(img)
        print(f"{tensor[:, 0, 0]}")
        #output_path = Path(__file__).with_name("test2.exr")
        #exr.pyexr.write(output_path, tensor.permute(1, 2, 0).numpy())
    else:
        print(f"${file_extension} files are currently not supported.")
        sys.exit()

    torch.manual_seed(1337)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device=device)
    train_network(tensor)


if __name__ == "__main__":
    main()
