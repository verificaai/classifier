"""
Extrai e cacheia embeddings do Qwen3-VL para todo o dataset organizado
(train/val/test), para depois treinar apenas a cabeça de classificação --
rápido e barato, sem precisar rodar o Qwen3-VL de novo a cada época.

A extração em si usa PyTorch/transformers (é como o Qwen3-VL está
disponível no Hugging Face), mas o resultado é salvo em .npz (numpy) puro
-- assim o restante do pipeline (modelo, treino, avaliação, todos em
TensorFlow/Keras) não precisa depender do PyTorch.

Exemplo:
    python src/extract_features.py \
        --data-dir dataset/humanos \
        --output-dir features/humanos
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from qwen_features import extract_embedding

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def extract_split(split_dir: Path, classes: list[str], model_id: str, use_4bit: bool):
    embeddings, labels = [], []
    for label_idx, class_name in enumerate(classes):
        class_dir = split_dir / class_name
        paths = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in VALID_EXT)
        for p in tqdm(paths, desc=f"{split_dir.name}/{class_name}"):
            image = Image.open(p).convert("RGB")
            emb = extract_embedding(image, model_id=model_id, use_4bit=use_4bit)
            embeddings.append(emb.numpy())
            labels.append(label_idx)

    return np.stack(embeddings).astype(np.float32), np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                         help="pasta dataset/<categoria> com train/val/test/real|fake")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                         help="default: Qwen3-VL-4B (cabe em GPU de 8GB com --model-id "
                              "Qwen/Qwen3-VL-4B-Instruct + 4-bit). Em GPU de 16GB+, pode "
                              "usar Qwen/Qwen3-VL-8B-Instruct para embeddings melhores.")
    parser.add_argument("--no-4bit", action="store_true",
                         help="desativa a quantização 4-bit (só use com GPU de 16GB+)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(p.name for p in (data_dir / "train").iterdir() if p.is_dir())
    print(f"Classes: {classes}")

    for split in ["train", "val", "test"]:
        embeddings, labels = extract_split(
            data_dir / split, classes, args.model_id, use_4bit=not args.no_4bit)
        np.savez(
            output_dir / f"{split}.npz",
            embeddings=embeddings, labels=labels, classes=np.array(classes),
            qwen_model_id=np.array(args.model_id),
            qwen_use_4bit=np.array(not args.no_4bit),
        )
        print(f"{split}: {embeddings.shape[0]} imagens, embedding_dim={embeddings.shape[1]}")

    print(f"\nFeatures salvas em: {output_dir}")


if __name__ == "__main__":
    main()
