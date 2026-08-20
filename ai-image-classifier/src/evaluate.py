"""
Avalia o modelo treinado no conjunto de teste: accuracy, matriz de confusão
e relatório de classificação (precision/recall/f1 por classe).

Exemplos:
    python src/evaluate.py --checkpoint checkpoints/humanos/qwen3vl \
        --features-dir features/humanos

    python src/evaluate.py --checkpoint checkpoints/humanos/resnet50 \
        --data-dir dataset/humanos
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

from dataset import build_image_dataset, load_feature_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                    help="pasta do checkpoint (contém best_model.keras + metadata.json)")
    p.add_argument("--features-dir", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)

    with open(checkpoint_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    classes = metadata["classes"]
    model_name = metadata["model_name"]

    model = keras.models.load_model(checkpoint_dir / "best_model.keras")

    if model_name == "qwen3vl":
        assert args.features_dir, "--features-dir é obrigatório para avaliar o modelo qwen3vl"
        test_ds, ds_classes, _, _ = load_feature_dataset(
            args.features_dir, "test", batch_size=args.batch_size)
    else:
        assert args.data_dir, "--data-dir é obrigatório para avaliar modelos baseados em imagem"
        use_freq_channel = metadata.get("use_freq_channel", metadata.get("in_channels", 3) == 4)
        test_ds, ds_classes = build_image_dataset(
            args.data_dir, "test", batch_size=args.batch_size, use_freq_channel=use_freq_channel)

    assert ds_classes == classes, "Classes do checkpoint não batem com o dataset/features atual."

    all_preds, all_labels = [], []
    for inputs, labels in test_ds:
        logits = model(inputs, training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    print("Matriz de confusão (linhas=real, colunas=predito):")
    print(confusion_matrix(all_labels, all_preds))
    print("\nRelatório de classificação:")
    print(classification_report(all_labels, all_preds, target_names=classes))


if __name__ == "__main__":
    main()
