import pyexr
import torch


def ParseEXR(filename) -> torch.Tensor:
    with pyexr.open(filename) as file:
        print(f"channels: {file.channels}, width: {file.width}, height: {file.height}")

        tensor = None
        # TODO: assumes only channels are R, G, B, A.
        # TODO: if all channels are the same, collapse into one
        for channel in file.channels:
            if tensor == None:
                tensor = torch.from_numpy(file.get(channel))
            else:
                tensor = torch.cat(
                    (tensor, torch.from_numpy(file.get(channel))), dim=-1
                )
        assert tensor.ndim == 3
        return tensor.permute(2, 1, 0)
