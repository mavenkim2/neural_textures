import torch


def random_crops_into(output: torch.tensor, input: torch.tensor, crop_dim: int):
    assert input.ndim == 3
    assert output.ndim == 4
    batch_size, channels, crop_h, crop_w = output.shape
    src_channels, height, width = input.shape

    assert channels == src_channels
    assert crop_h == crop_dim
    assert crop_w == crop_dim
    assert crop_dim <= height
    assert crop_dim <= width

    width, height = input.shape[2], input.shape[1]
    for batch_index in range(batch_size):
        start_u = torch.randint(0, width - crop_dim + 1, (1,)).item()
        start_v = torch.randint(0, height - crop_dim + 1, (1,)).item()
        output[batch_index].copy_(
            input[:, start_v : start_v + crop_dim, start_u : start_u + crop_dim]
        )
