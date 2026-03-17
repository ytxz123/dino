import argparse
import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    if __package__:
        return

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_import_path()

from three_class_segmentation.config import load_config
from three_class_segmentation.engine import evaluate, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv3 ViT-L/16 语义分割")
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="运行模式")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    if args.mode == "train":
        train(config)
        return
    evaluate(config)


if __name__ == "__main__":
    main()