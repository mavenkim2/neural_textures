import pytest
import torch
from transformer import exr
from transformer import utils
from transformer import model
from pathlib import Path


@pytest.fixture
def batch_tensor(request, batch_size=4, crop_dim=256):
    torch.manual_seed(1337)

    exr_file = request.param
    exr_path = Path(__file__).with_name(exr_file)
    image = exr.ParseEXR(exr_path)

    channels, width, height = image.shape[0], image.shape[2], image.shape[1]

    assert crop_dim <= width and crop_dim <= height

    batch_tensor = image.new_empty((batch_size, channels, crop_dim, crop_dim))
    utils.random_crops_into(batch_tensor, image, crop_dim)

    return batch_tensor


@pytest.mark.parametrize("batch_tensor", ["out.exr"], indirect=True)
def test_crops(batch_tensor):

    batch_size = batch_tensor.shape[0]

    for i in range(batch_size):
        crop_path = Path(__file__).with_name(f"crop{i}.exr")
        exr.pyexr.write(crop_path, batch_tensor[i].permute(2, 1, 0).numpy())
    assert True


@pytest.mark.parametrize("batch_tensor", ["out.exr"], indirect=True)
def test_global_transformation(batch_tensor):

    batch_size, channels, crop_dim, _ = batch_tensor.shape

    network = model.CompressionNetwork(channels)
    encoded_tensor = network.global_transformation(batch_tensor)

    assert encoded_tensor.shape == (
        batch_size,
        network.channels_m,
        crop_dim / 8,
        crop_dim / 8,
    ), f"Unexpected encoded_tensor shape: {encoded_tensor.shape}"
    assert torch.all(encoded_tensor >= -0.5)
    assert torch.all(encoded_tensor <= 0.5)
