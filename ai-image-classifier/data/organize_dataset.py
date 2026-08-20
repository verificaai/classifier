"""
Organiza um dataset na estrutura esperada
pelo keras.utils.image_dataset_from_directory:

dataset/<categoria>/train/real|fake
dataset/<categoria>/val/real|fake
dataset/<categoria>/test/real|fake

Exemplo:
    python data/organize_dataset.py --category humanos \
        --source-real cropped/humanos/real --source-fake cropped/humanos/fake
"""
import argparse
import random
import shutil
from pathlib import Path

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 123

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXT]


def split_and_copy(files, label: str, output_dir: Path):
    random.Random(SEED).shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    for split_name, split_files in splits.items():
        out_dir = output_dir / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy2(f, out_dir / f.name)

    print(f"[{label}] total={n} train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--source-real", type=str, default=None,
                         help="default: cropped/<categoria>/real")
    parser.add_argument("--source-fake", type=str, default=None,
                         help="default: cropped/<categoria>/fake")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="default: dataset/<categoria>")
    args = parser.parse_args()

    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6

    source_real = Path(args.source_real or f"cropped/{args.category}/real")
    source_fake = Path(args.source_fake or f"cropped/{args.category}/fake")
    output_dir = Path(args.output_dir or f"dataset/{args.category}")

    real_files = list_images(source_real)
    fake_files = list_images(source_fake)

    if not real_files or not fake_files:
        raise SystemExit(
            f"Nenhuma imagem encontrada em {source_real} ou {source_fake}. "
            "Rode src/preprocess_faces.py antes, ou ajuste --source-real/--source-fake."
        )

    split_and_copy(real_files, "real", output_dir)
    split_and_copy(fake_files, "fake", output_dir)

    print(f"\nDataset organizado em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
