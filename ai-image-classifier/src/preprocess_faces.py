"""
Aplica pré-processamento em lote nas pastas de imagens reais e artificiais,
ANTES de rodar data/organize_dataset.py. O comportamento (recorte facial ou
não) é definido automaticamente pela categoria em src/categories.py.

Fluxo completo recomendado:
    1. raw_data/<categoria>/real/*        (fotos reais originais)
    2. raw_data/<categoria>/fake/*        (saída de generate_fake_images.py)
    3. python src/preprocess_faces.py --category humanos \
           --real-dir raw_data/humanos/real --fake-dir raw_data/humanos/fake
       (saída default: cropped/<categoria>/real e cropped/<categoria>/fake)
    4. python data/organize_dataset.py --category humanos
"""
import argparse

from categories import get_category_config
from face_utils import process_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--fake-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None,
                         help="default: cropped/<categoria>")
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    config = get_category_config(args.category)
    apply_crop = config["apply_face_crop"]
    print(f"Categoria: {args.category} | recorte facial: {apply_crop}")

    output_dir = args.output_dir or f"cropped/{args.category}"

    print("Processando imagens reais...")
    process_folder(args.real_dir, f"{output_dir}/real", margin=args.margin,
                    output_size=args.size, apply_crop=apply_crop)

    print("Processando imagens artificiais...")
    process_folder(args.fake_dir, f"{output_dir}/fake", margin=args.margin,
                    output_size=args.size, apply_crop=apply_crop)

    print(f"\nSaída em: {output_dir}")


if __name__ == "__main__":
    main()
