import torch

from rainmap_nowcasting.model import RainMapUNet


def test_model_output_shape_matches_target_frames() -> None:
    model = RainMapUNet(input_frames=6, target_frames=6, base_channels=8)
    x = torch.rand(2, 6, 65, 67)
    y = model(x)

    assert y.shape == (2, 6, 65, 67)
    assert torch.all(y >= 0)
    assert torch.all(y <= 1)
