from pathlib import Path

from PIL import Image
from safetensors.torch import save_file

from rainmap_nowcasting.config import write_json
from rainmap_nowcasting.hf import ModelFiles
from rainmap_nowcasting.inference import predict_frames
from rainmap_nowcasting.model import RainMapUNet


def test_predict_frames_writes_pngs_and_gif(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = RainMapUNet(input_frames=6, target_frames=6, base_channels=8)
    save_file(model.state_dict(), str(model_dir / "rainmap-nowcasting-demo.safetensors"))
    write_json(
        model_dir / "model_config.json",
        {
            "model_type": "RainMapUNet",
            "input_frames": 6,
            "target_frames": 6,
            "image_size": [32, 32],
            "base_channels": 8,
        },
    )

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(6):
        Image.new("L", (32, 32), index * 20).save(input_dir / f"frame_{index:03d}.png")

    output_dir = tmp_path / "output"
    metadata = predict_frames(
        input_dir=input_dir,
        output_dir=output_dir,
        model_files=ModelFiles(
            model_dir=model_dir,
            weights_path=model_dir / "rainmap-nowcasting-demo.safetensors",
            config_path=model_dir / "model_config.json",
            repo_id="local-test",
        ),
        device_name="cpu",
    )

    assert len(metadata["frame_paths"]) == 6
    assert (output_dir / "prediction_000.png").exists()
    assert (output_dir / "prediction.gif").exists()
    assert (output_dir / "prediction_metadata.json").exists()
