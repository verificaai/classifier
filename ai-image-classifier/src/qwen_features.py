"""
Extrai embeddings visuais do Qwen3-VL para usar como features de transfer
learning no classificador, em vez de treinar/fine-tunar uma CNN.

Estratégia: passamos a imagem pelo Qwen3-VL com um prompt mínimo e pegamos
os hidden states da última camada, fazendo mean-pooling sobre as posições
correspondentes aos tokens de imagem (identificados por
`model.config.image_token_id`). Isso usa apenas a API pública do
`model.forward()` / `output_hidden_states=True`, então não depende de nomes
internos de submódulos que podem mudar entre versões do Qwen3-VL.

Nota: confirme `model.config.image_token_id` na versão do transformers/Qwen3-VL
que você instalar (ex: `print(model.config)`) -- se o nome do atributo mudar,
ajuste a função `_get_image_token_id` abaixo.

Nota sobre VRAM: em GPU de 8 GB, use o Qwen3-VL-4B-Instruct com
use_4bit=True (defaults abaixo) -- o 8B não cabe nem quantizado (~12 GB).
Com GPU maior (16 GB+), pode trocar para Qwen/Qwen3-VL-8B-Instruct.
"""
from __future__ import annotations

import torch
from PIL import Image

_model = None
_processor = None
_model_id = None


def load_qwen(model_id: str = "Qwen/Qwen3-VL-4B-Instruct", use_4bit: bool = True):
    global _model, _processor, _model_id
    if _model is not None and _model_id == model_id:
        return _model, _processor

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Carregando {model_id} (4-bit={use_4bit})...")
    _processor = AutoProcessor.from_pretrained(model_id)

    if use_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=quantization_config, device_map="auto"
        )
    else:
        _model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    _model.eval()
    _model_id = model_id
    return _model, _processor


def _get_image_token_id(model) -> int | None:
    for attr in ("image_token_id", "image_token_index"):
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    return None


@torch.no_grad()
def extract_embedding(image: Image.Image, model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
                       use_4bit: bool = True) -> torch.Tensor:
    """Retorna um vetor 1D representando a imagem (embedding do Qwen3-VL)."""
    model, processor = load_qwen(model_id, use_4bit=use_4bit)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the image."},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]

    image_token_id = _get_image_token_id(model)
    if image_token_id is not None:
        mask = inputs["input_ids"][0] == image_token_id
        if mask.any():
            return last_hidden[mask].mean(dim=0).float().cpu()

    # fallback: se não achar o id do token de imagem, usa a média de tudo.
    # Isso dilui o embedding com os tokens do prompt de texto -- se você ver
    # este aviso, confira o nome do atributo em model.config (ver docstring
    # do módulo) antes de treinar com esses embeddings.
    print("Aviso: image_token_id não encontrado -- usando média de todos os "
          "tokens (embedding pode estar diluído com o prompt de texto).")
    return last_hidden.mean(dim=0).float().cpu()
