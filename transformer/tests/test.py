import pytest
import exr
import torch


@pytest.mark.parameterize("exr_file", "out.exr")
def test_crops(exr_file):
    crop_dim = 256
    image = exr.ParseEXR(exr_file)

    width, height = image.shape[2], image.shape[1]
    start_u = torch.randint(0, width - crop_dim + 1)
    start_v = torch.randint(0, height - crop_dim + 1)
    out = image[:, start_v : start_v + crop_dim, start_u : start_u + crop_dim]

    exr.pyexr.write("crop.exr", out.permute(2, 1, 0))
    assert True
