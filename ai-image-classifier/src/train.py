"""
Treina o classificador real vs. fake em TensorFlow/Keras.

Loop de treino customizado com tf.GradientTape (em vez de model.fit()),
seguindo o padrão do projeto de referência: early stopping por paciência,
salva o melhor checkpoint e plota curvas de accuracy/loss ao final.

Backbone recomendado: qwen3vl (transfer learning sobre embeddings
pré-extraídos -- rode src/extract_features.py antes).

Exemplos:
    # Qwen3-VL (recomendado)
    python src/extract_features.py --data-dir dataset/humanos --output-dir features/humanos
    python src/train.py --category humanos --model qwen3vl --features-dir features/humanos --epochs 30

    # CNN com transfer learning (alternativa mais leve, sem GPU grande)
    python src/train.py --category humanos --model resnet50 --data-dir dataset/humanos --epochs 15
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from dataset import build_image_dataset, load_feature_dataset
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", type=str, required=True,
                    help="nome da categoria (ex: humanos) -- usado para organizar checkpoints")
    p.add_argument("--model", type=str, default="qwen3vl",
                    choices=["qwen3vl", "simple_cnn", "resnet50", "efficientnet_b0"])
    p.add_argument("--features-dir", type=str, default=None,
                    help="pasta com train/val/test.npz (obrigatório se --model qwen3vl)")
    p.add_argument("--data-dir", type=str, default=None,
                    help="pasta dataset/<categoria> com imagens (obrigatório para os outros modelos)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="default: checkpoints/<categoria>/<modelo>")
    p.add_argument("--no-freq-channel", action="store_true",
                    help="(só para CNN) desativa o canal extra de resíduo de alta frequência")
    p.add_argument("--freeze-backbone", action="store_true",
                    help="(só para resnet50/efficientnet_b0) congela a backbone pré-treinada, "
                         "treinando só a cabeça de classificação")
    return p.parse_args()


def run_epoch(model, dataset, loss_fn, optimizer, train: bool):
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(dataset, desc="train" if train else "val"):
        if train:
            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                loss = loss_fn(labels, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            logits = model(inputs, training=False)
            loss = loss_fn(labels, logits)

        batch_size = int(tf.shape(labels)[0])
        total_loss += float(loss) * batch_size
        preds = tf.argmax(logits, axis=1, output_type=labels.dtype)
        correct += int(tf.reduce_sum(tf.cast(preds == labels, tf.int32)))
        total += batch_size

    return total_loss / total, correct / total


def main():
    args = parse_args()
    print(f"Categoria: {args.category} | modelo: {args.model}")

    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/{args.category}/{args.model}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    embedding_dim = None
    in_channels = 3
    use_freq_channel = not args.no_freq_channel

    qwen_info = None
    if args.model == "qwen3vl":
        assert args.features_dir, "--features-dir é obrigatório para --model qwen3vl"
        train_ds, classes, embedding_dim, qwen_info = load_feature_dataset(
            args.features_dir, "train", batch_size=args.batch_size)
        val_ds, _, _, _ = load_feature_dataset(
            args.features_dir, "val", batch_size=args.batch_size)
    else:
        assert args.data_dir, "--data-dir é obrigatório para modelos baseados em imagem"
        in_channels = 4 if use_freq_channel else 3
        train_ds, classes = build_image_dataset(
            args.data_dir, "train", batch_size=args.batch_size, use_freq_channel=use_freq_channel)
        val_ds, _ = build_image_dataset(
            args.data_dir, "val", batch_size=args.batch_size, use_freq_channel=use_freq_channel)

    print(f"Classes: {classes}")  # esperado: ['fake', 'real']

    model = build_model(args.model, num_classes=len(classes), in_channels=in_channels,
                         embedding_dim=embedding_dim, freeze_backbone=args.freeze_backbone)
    model.summary()

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = run_epoch(model, train_ds, loss_fn, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, val_ds, loss_fn, optimizer, train=False)

        print(f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0

            model.save(checkpoint_dir / "best_model.keras")
            metadata = {
                "model_name": args.model,
                "classes": classes,
                "in_channels": in_channels,
                "use_freq_channel": in_channels == 4,
                "embedding_dim": embedding_dim,
                "category": args.category,
            }
            if qwen_info is not None:
                # inference.py/evaluate.py precisam disso para extrair
                # embeddings com o MESMO modelo/config usado no treino --
                # senão o embedding não bate com o que a cabeça aprendeu.
                metadata.update(qwen_info)
            with open(checkpoint_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print("-> novo melhor modelo salvo")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    plot_history(history, checkpoint_dir / "training_curves.png")


def plot_history(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_acc"], label="train")
    axes[0].plot(history["val_acc"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.savefig(output_path, dpi=150)
    print(f"Curvas de treino salvas em: {output_path}")


if __name__ == "__main__":
    main()
