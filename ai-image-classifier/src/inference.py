"""
Classifica uma imagem OU um vídeo como real ou gerado por IA.

Para vídeo: extrai frames em intervalos regulares, classifica cada frame
como se fosse uma imagem, e agrega os resultados (média das probabilidades
softmax de todos os frames, não só o voto majoritário -- assim um frame
com confiança alta pesa mais que um frame incerto) num veredito final.
Também reporta o % de frames que concordam com o veredito, como sinal de
confiabilidade: se os frames discordam muito entre si, o resultado deve
ser tratado com cautela.

Limitação importante (o classificador não foi refeito para vídeo, só a
inferência): o modelo foi treinado com fotos estáticas reais vs. imagens
estáticas geradas por IA. Aplicá-lo frame a frame é uma primeira
abordagem razoável, mas artefatos de compressão de vídeo (H.264/H.265) são
diferentes dos artefatos de uma foto crua -- isso pode derrubar a acurácia
em vídeo comprimido (ex: vídeo baixado do WhatsApp/Instagram). Se a
acurácia em vídeo se mostrar baixa na prática, o próximo passo seria
incluir frames extraídos de vídeos reais (não só fotos) como exemplos
"real" no dataset de treino, para o classificador aprender a distinguir
artefato de compressão de artefato de geração por IA.

Exemplos:
    python src/inference.py --image foto.jpg --checkpoint checkpoints/humanos/qwen3vl
    python src/inference.py --video clipe.mp4 --checkpoint checkpoints/humanos/qwen3vl
    python src/inference.py --video clipe.mp4 --checkpoint checkpoints/humanos/qwen3vl \
        --frame-interval 0.5 --max-frames 60
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from dataset import preprocess_single_image
from face_utils import detect_and_crop_face


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="caminho de uma imagem")
    src.add_argument("--video", type=str, help="caminho de um vídeo (mp4, mov, avi, ...)")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="pasta do checkpoint (contém best_model.keras + metadata.json)")
    p.add_argument("--no-face-crop", action="store_true",
                    help="pula a detecção/recorte facial e usa a imagem/frame inteiro")
    p.add_argument("--frame-interval", type=float, default=1.0,
                    help="[vídeo] segundos entre frames amostrados (default: 1.0)")
    p.add_argument("--max-frames", type=int, default=30,
                    help="[vídeo] número máximo de frames a analisar (default: 30 -- "
                         "cada frame roda uma inferência completa, então isso limita o "
                         "tempo total, especialmente com --model qwen3vl)")
    return p.parse_args()


def load_checkpoint(checkpoint_dir: Path):
    with open(checkpoint_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    model = keras.models.load_model(checkpoint_dir / "best_model.keras")
    return model, metadata


def classify_pil_image(image: Image.Image, model, metadata: dict, no_face_crop: bool):
    """Classifica uma única imagem PIL. Retorna (classes, probs), com probs
    alinhado a metadata['classes']."""
    classes = metadata["classes"]
    model_name = metadata["model_name"]

    if not no_face_crop:
        face = detect_and_crop_face(image)
        image = face if face is not None else image

    if model_name == "qwen3vl":
        from qwen_features import extract_embedding

        # Mesmo modelo/config do Qwen3-VL usado no treino (ver train.py) --
        # senão o embedding não bate com o que a cabeça aprendeu.
        qwen_model_id = metadata.get("qwen_model_id", "Qwen/Qwen3-VL-4B-Instruct")
        qwen_use_4bit = metadata.get("qwen_use_4bit", True)
        embedding = extract_embedding(
            image, model_id=qwen_model_id, use_4bit=qwen_use_4bit
        ).numpy()[None, :]
        logits = model(embedding, training=False)
    else:
        use_freq_channel = metadata.get("use_freq_channel", metadata.get("in_channels", 3) == 4)
        array = preprocess_single_image(image, use_freq_channel=use_freq_channel)
        logits = model(array, training=False)

    probs = tf.nn.softmax(logits, axis=1)[0].numpy()
    return classes, probs


def iter_video_frames(video_path: str, frame_interval: float, max_frames: int):
    """Gera (timestamp_segundos, PIL.Image) amostrados a cada `frame_interval`
    segundos, até no máximo `max_frames` frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback se o metadado do vídeo faltar
    step = max(1, round(fps * frame_interval))

    frame_idx, yielded = 0, 0
    while yielded < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_idx / fps, Image.fromarray(frame_rgb)
            yielded += 1
        frame_idx += 1
    cap.release()


def classify_video(video_path: str, model, metadata: dict, no_face_crop: bool,
                    frame_interval: float, max_frames: int):
    classes = metadata["classes"]
    per_frame_probs = []

    for timestamp, frame in iter_video_frames(video_path, frame_interval, max_frames):
        _, probs = classify_pil_image(frame, model, metadata, no_face_crop)
        per_frame_probs.append(probs)
        pred = classes[int(probs.argmax())]
        print(f"  t={timestamp:5.1f}s  {pred:>10s}  ({probs.max()*100:5.1f}%)")

    if not per_frame_probs:
        raise SystemExit("Nenhum frame pôde ser extraído do vídeo -- confira o caminho/formato.")

    all_probs = np.stack(per_frame_probs)
    mean_probs = all_probs.mean(axis=0)
    final_idx = int(mean_probs.argmax())
    agreement = float((all_probs.argmax(axis=1) == final_idx).mean())

    return classes, mean_probs, len(per_frame_probs), agreement


def main():
    args = parse_args()
    model, metadata = load_checkpoint(Path(args.checkpoint))

    if args.image:
        image = Image.open(args.image).convert("RGB")
        classes, probs = classify_pil_image(image, model, metadata, args.no_face_crop)
        pred_idx = int(probs.argmax())
        print(f"Predição: {classes[pred_idx]} (confiança: {probs[pred_idx]*100:.2f}%)")
        for c, p in zip(classes, probs.tolist()):
            print(f"  {c}: {p*100:.2f}%")
        return

    print(f"Analisando frames de {args.video} "
          f"(1 a cada {args.frame_interval}s, até {args.max_frames} frames)...")
    classes, mean_probs, n_frames, agreement = classify_video(
        args.video, model, metadata, args.no_face_crop,
        args.frame_interval, args.max_frames)

    pred_idx = int(mean_probs.argmax())
    print(f"\n{n_frames} frames analisados | {agreement*100:.0f}% concordam com o veredito final")
    print(f"Predição do vídeo: {classes[pred_idx]} (confiança média: {mean_probs[pred_idx]*100:.2f}%)")
    for c, p in zip(classes, mean_probs.tolist()):
        print(f"  {c}: {p*100:.2f}%")

    if agreement < 0.7:
        print("\nAviso: os frames discordam bastante entre si -- o vídeo pode ter "
              "trechos reais e trechos gerados por IA, ou o resultado é pouco confiável. "
              "Vale revisar os frames individuais listados acima.")


if __name__ == "__main__":
    main()
