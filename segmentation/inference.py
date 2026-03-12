import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dinov3.segmentation.dataset import image_to_tensor, load_rgb_image, prediction_to_png
from dinov3.segmentation.model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone-weights", type=str, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.png"))
    return [input_path]


class InferenceImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], size: int):
        self.image_paths = image_paths
        self.size = size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]
        image = load_rgb_image(image_path, size=self.size)
        return image_to_tensor(image), image_path.name


def save_prediction_batch(predictions: torch.Tensor, file_names: list[str], output_path: Path) -> None:
    for prediction, file_name in zip(predictions, file_names):
        prediction_to_png(prediction).save(output_path / file_name)


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
        dataset = InferenceImageDataset(input_files, size=args.size)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        for tensors, file_names in dataloader:
            tensors = tensors.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions = model(tensors).argmax(dim=1)
            save_prediction_batch(predictions.cpu(), list(file_names), output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_path = input_files[0]
        image = load_rgb_image(image_path, size=args.size)
        tensor = image_to_tensor(image).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(tensor).argmax(dim=1).squeeze(0)
        prediction_to_png(prediction).save(output_path)


if __name__ == "__main__":
    main()