"""
Pega N imagens aleatorias de uma pasta de dataset já existente para dentro de raw_data/<categoria>/real.

"""
import argparse
import random
import shutil
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True,
                         help="Pasta do dataset original (pode ter subpastas, ex: img_align_celeba/)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", type=str, default="copy",
                         choices=["copy", "symlink"],
                         help="'symlink' economiza espaço/tempo, mas exige que o "
                              "Drive continue montado quando os scripts seguintes rodarem")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Procurando imagens em {source_dir} (pode demorar se for uma pasta grande no Drive)...")
    all_images = [p for p in source_dir.rglob("*") if p.suffix.lower() in VALID_EXT]

    if not all_images:
        raise SystemExit(f"Nenhuma imagem encontrada em {source_dir}")

    print(f"Total de imagens encontradas: {len(all_images)}")

    n = min(args.n_samples, len(all_images))
    random.Random(args.seed).shuffle(all_images)
    sampled = all_images[:n]

    for p in sampled:
        dest = output_dir / p.name
        if args.copy_mode == "symlink":
            if not dest.exists():
                dest.symlink_to(p.resolve())
        else:
            shutil.copy2(p, dest)

    print(f"{n} imagens {'copiadas' if args.copy_mode == 'copy' else 'linkadas'} para: {output_dir}")


if __name__ == "__main__":
    main()
