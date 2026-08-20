"""
usa Qwen3-VL para gerar descrições das imagens reais
da categoria escolhida (ex: humanos). Essas descrições serão usadas
para gerar as versões artificiais com Qwen-Image / FLUX 2.

O prompt de legenda é definido por categoria em src/categories.py -- assim,
categorias diferentes (ex: paisagens, animais) podem ter um prompt mais
apropriado sem precisar tocar neste script.

Saída: um JSONL com {"image_path": ..., "caption": ...} por linha, que o
script generate_fake_images.py vai consumir.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from categories import get_category_config  # noqa: E402

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_model(model_id: str = "Qwen/Qwen3-VL-4B-Instruct", use_4bit: bool = True):
    # Import tardio para não exigir 'transformers' em quem só quer treinar o classificador
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)

    if use_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=quantization_config, device_map="auto"
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    return model, processor


def caption_image(model, processor, image_path: Path, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=120)

    output_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]

    return output_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True,
                         help="ex: humanos -- define o prompt de legenda (ver src/categories.py)")
    parser.add_argument("--input-dir", type=str, default=None,
                         help="default: raw_data/<categoria>/real")
    parser.add_argument("--output-file", type=str, default=None,
                         help="default: captions_<categoria>.jsonl")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--no-4bit", action="store_true",
                         help="desativa a quantização 4-bit (só use com GPU de 16GB+ "
                              "ou se --model-id for um modelo pequeno)")
    args = parser.parse_args()

    config = get_category_config(args.category)
    prompt = config["caption_prompt"]

    input_dir = Path(args.input_dir or f"raw_data/{args.category}/real")
    output_path = Path(args.output_file or f"captions_{args.category}.jsonl")

    image_paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXT)
    if not image_paths:
        raise SystemExit(f"Nenhuma imagem encontrada em {input_dir}")

    print(f"Carregando Qwen3-VL ({args.model_id}, 4-bit={not args.no_4bit})...")
    model, processor = load_model(args.model_id, use_4bit=not args.no_4bit)

    already_done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            already_done = {json.loads(line)["image_path"] for line in f}

    with open(output_path, "a", encoding="utf-8") as f:
        for img_path in tqdm(image_paths, desc="Gerando legendas"):
            if str(img_path) in already_done:
                continue
            try:
                caption = caption_image(model, processor, img_path, prompt)
            except Exception as e:
                print(f"Falha em {img_path}: {e}")
                continue

            f.write(json.dumps({"image_path": str(img_path), "caption": caption},
                                ensure_ascii=False) + "\n")
            f.flush()

    print(f"Legendas salvas em: {output_path}")


if __name__ == "__main__":
    main()
