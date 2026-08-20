"""
Detecção, alinhamento e recorte de rostos.

Usa o pacote `mtcnn` (implementação em TensorFlow/Keras), para manter o
pipeline inteiro em TensorFlow. Para cada imagem, detecta o maior rosto,
recorta com uma margem e devolve uma imagem quadrada -- assim o classificador
recebe uma entrada padronizada, focada no rosto, em vez de fotos inteiras
com enquadramentos e distâncias variadas.

Nota de compatibilidade: o pacote `mtcnn` já teve mudanças de API entre
versões. Este código foi escrito para a versão atual (retorna uma lista de
dicts com chave "box" no formato [x, y, largura, altura]). Se você instalar
uma versão mais antiga e vir um erro de KeyError/formato, rode
`pip install --upgrade mtcnn` ou ajuste `_extract_box` abaixo conforme a
saída de `detector.detect_faces(...)` na sua versão.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector


def _extract_box(face: dict) -> tuple[float, float, float, float]:
    """Extrai (x1, y1, x2, y2) do dict retornado por detect_faces."""
    x, y, w, h = face["box"]
    return float(x), float(y), float(x + w), float(y + h)


def detect_and_crop_face(image: Image.Image, margin: float = 0.4,
                          output_size: int = 224) -> Image.Image | None:
    """
    Retorna a imagem recortada e redimensionada com foco no rosto, ou None
    se nenhum rosto for detectado (nesse caso, o chamador deve decidir se
    descarta a imagem ou usa a imagem original como fallback).
    """
    detector = _get_detector()
    image_rgb = image.convert("RGB")
    faces = detector.detect_faces(np.array(image_rgb))

    if not faces:
        return None

    # pega o maior rosto (por área da caixa), caso haja mais de um na foto
    faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    x1, y1, x2, y2 = _extract_box(faces[0])
    w, h = x2 - x1, y2 - y1

    # adiciona margem ao redor do rosto detectado
    x1 -= w * margin
    y1 -= h * margin
    x2 += w * margin
    y2 += h * margin

    img_w, img_h = image_rgb.size
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

    face_crop = image_rgb.crop((x1, y1, x2, y2))
    face_crop = face_crop.resize((output_size, output_size), Image.BICUBIC)
    return face_crop


def process_folder(input_dir: str, output_dir: str, margin: float = 0.4,
                    output_size: int = 224, fallback_to_original: bool = False,
                    apply_crop: bool = True):
    """Aplica detecção+recorte facial a todas as imagens de uma pasta.

    Se apply_crop=False (categorias que não são de rosto/humanos), apenas
    redimensiona e copia as imagens, sem rodar o detector de rosto."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in valid_ext)

    n_ok, n_failed = 0, 0
    for p in paths:
        try:
            image = Image.open(p).convert("RGB")

            if not apply_crop:
                image.resize((output_size, output_size), Image.BICUBIC).save(output_dir / p.name, quality=95)
                n_ok += 1
                continue

            face = detect_and_crop_face(image, margin=margin, output_size=output_size)

            if face is None:
                n_failed += 1
                if not fallback_to_original:
                    continue
                face = image.resize((output_size, output_size), Image.BICUBIC)

            face.save(output_dir / p.name, quality=95)
            n_ok += 1
        except Exception as e:
            print(f"Falha em {p}: {e}")
            n_failed += 1

    print(f"[{input_dir.name}] processados: {n_ok}, falhas/sem rosto: {n_failed}")
