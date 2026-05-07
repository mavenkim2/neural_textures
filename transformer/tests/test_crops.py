import pytest
import torch
from transformer import exr
from transformer import utils
from pathlib import Path


@pytest.mark.parametrize("exr_file", ["out.exr"])
def test_crops(exr_file):
    torch.manual_seed(1337)
    crop_dim = 256
    exr_path = Path(__file__).with_name(exr_file)
    image = exr.ParseEXR(exr_path)

    batch_size = 4
    channels, width, height = image.shape[0], image.shape[2], image.shape[1]

    batch_tensor = image.new_empty((batch_size, channels, crop_dim, crop_dim))
    utils.random_crops_into(batch_tensor, image, crop_dim)

    for i in range(batch_size):
        crop_path = Path(__file__).with_name(f"crop{i}.exr")
        exr.pyexr.write(crop_path, batch_tensor[i].permute(2, 1, 0).numpy())
    assert True
