import argparse
from pathlib import Path

import torch

from dinov3.segmentation.dataset import image_to_tensor, load_rgb_image, prediction_to_png
from dinov3.segmentation.model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone-weights", type=str, default=None)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.png"))
    return [input_path]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint, override_backbone_weights=args.backbone_weights).to(device)
    model.eval()

    input_path = Path(args.input)
    output_path = Path(args.output)
    input_files = iter_input_files(input_path)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for image_path in input_files:
        image = load_rgb_image(image_path, size=args.size)
        tensor = image_to_tensor(image).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(tensor).argmax(dim=1).squeeze(0)
        mask = prediction_to_png(prediction)
        if input_path.is_dir():
            mask.save(output_path / image_path.name)
        else:
            mask.save(output_path)


if __name__ == "__main__":
    main()