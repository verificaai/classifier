"""
------------------------------------------------------------------------
TROCAR DE PROVEDOR (DashScope <-> fal.ai): só mexa nas duas linhas abaixo,
nada mais no resto do arquivo precisa mudar.

Opção 1 -- DashScope (Alibaba Cloud, oficial do Qwen-Image):
    1. pip install dashscope
    2. Pegue uma API key em https://www.alibabacloud.com/help/en/model-studio/get-api-key
    3. Defina a variável de ambiente:
         Linux/Mac:  export DASHSCOPE_API_KEY="sk-xxxxxxxx"
         Windows:    set DASHSCOPE_API_KEY=sk-xxxxxxxx
    4. Rode com: --api-provider dashscope   (já é o default)

Opção 2 -- fal.ai (terceiro, cadastro mais simples):
    1. pip install fal-client
    2. Pegue uma API key em https://fal.ai/dashboard/keys
    3. Defina a variável de ambiente:
         Linux/Mac:  export FAL_KEY="xxxxxxxx"
         Windows:    set FAL_KEY=xxxxxxxx
    4. Rode com: --api-provider fal
------------------------------------------------------------------------
"""
import argparse
import json
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# Modelo default de cada provedor para a categoria "qwen-image". 
# testar outro modelo no mesmo provedor (ex: um FLUX hospedado
# no fal.ai), é só passar --model-id na linha de comando -- não precisa
# editar este dicionário.
DEFAULT_MODEL_ID = {
    "dashscope": "qwen-image-plus",
    "fal": "fal-ai/qwen-image",
}


def generate_image_dashscope(prompt: str, model_id: str, seed: int | None = None) -> bytes:
    import os
    from http import HTTPStatus
    from dashscope import ImageSynthesis

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variável de ambiente DASHSCOPE_API_KEY não definida. Veja o "
            "cabeçalho deste arquivo para instruções."
        )

    kwargs = {"seed": seed} if seed is not None else {}
    rsp = ImageSynthesis.call(
        api_key=api_key,
        model=model_id,
        prompt=prompt,
        negative_prompt=" ",
        n=1,
        size="1328*1328",
        prompt_extend=True,
        watermark=False,
        **kwargs,
    )

    if rsp.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"DashScope falhou: status={rsp.status_code} code={rsp.code} "
            f"message={rsp.message}"
        )

    image_url = rsp.output.results[0].url
    return requests.get(image_url, timeout=60).content


def generate_image_fal(prompt: str, model_id: str, seed: int | None = None) -> bytes:
    """Gera uma imagem via fal.ai e retorna os bytes da imagem."""
    import fal_client

    arguments = {"prompt": prompt}
    if seed is not None:
        arguments["seed"] = seed

    result = fal_client.subscribe(model_id, arguments=arguments)
    image_url = result["images"][0]["url"]
    return requests.get(image_url, timeout=60).content


# Registro de provedores disponíveis -- para adicionar um terceiro provedor
# no futuro (ex: Replicate), basta escrever uma função generate_image_xxx(
# prompt, model_id, seed) -> bytes e adicionar uma entrada aqui.
PROVIDERS = {
    "dashscope": generate_image_dashscope,
    "fal": generate_image_fal,
}


def generate_image(provider: str, model_id: str, prompt: str, seed: int | None = None) -> bytes:
    return PROVIDERS[provider](prompt, model_id, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--captions", type=str, default=None,
                         help="default: captions_<categoria>.jsonl")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="default: raw_data/<categoria>/fake")
    parser.add_argument("--api-provider", type=str, default="dashscope",
                         choices=list(PROVIDERS.keys()))
    parser.add_argument("--model-id", type=str, default=None,
                         help="default: qwen-image-plus (dashscope) ou "
                              "fal-ai/qwen-image (fal)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3,
                         help="tentativas por imagem antes de desistir dela "
                              "(erros de rede/rate limit são comuns em API)")
    args = parser.parse_args()

    model_id = args.model_id or DEFAULT_MODEL_ID[args.api_provider]

    captions_path = args.captions or f"captions_{args.category}.jsonl"
    output_dir = Path(args.output_dir or f"raw_data/{args.category}/fake")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(captions_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    print(f"Provedor: {args.api_provider} | modelo: {model_id}")

    n_ok, n_failed = 0, 0
    for i, entry in enumerate(tqdm(entries, desc="Gerando imagens")):
        original_name = Path(entry["image_path"]).stem
        out_path = output_dir / f"{original_name}_fake.jpg"
        if out_path.exists():
            continue

        seed = args.seed if args.seed is not None else i

        for attempt in range(args.max_retries):
            try:
                image_bytes = generate_image(
                    args.api_provider, model_id, entry["caption"], seed=seed)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image.save(out_path, quality=95)
                n_ok += 1
                break
            except Exception as e:
                is_last = attempt + 1 == args.max_retries
                print(f"[{original_name}] tentativa {attempt + 1}/{args.max_retries} "
                      f"falhou: {e}")
                if is_last:
                    n_failed += 1
                else:
                    time.sleep(2 ** attempt)  # backoff exponencial (rate limit)

    print(f"\nGeradas: {n_ok} | falhas: {n_failed}")
    print(f"Imagens artificiais salvas em: {output_dir}")


if __name__ == "__main__":
    main()
