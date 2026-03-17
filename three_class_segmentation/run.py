import argparse

from .config import load_config
from .engine import evaluate, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv3 ViT-L/16 三类语义分割")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="运行模式")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "train":
        train(config)
        return
    evaluate(config)


if __name__ == "__main__":
    main()