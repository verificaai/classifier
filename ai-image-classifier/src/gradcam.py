"""
Grad-CAM em TensorFlow para visualizar quais regiões da imagem o modelo usou
para decidir "real" ou "fake". Importante para validar que o modelo está
aprendendo padrões do rosto (pele, olhos, dentes, contorno do cabelo) e não
algum viés espúrio do dataset (fundo, watermark, resolução).

Não se aplica ao modelo qwen3vl (embeddings já global-pooled do Qwen3-VL,
sem mapa espacial de ativações) -- só aos backbones baseados em imagem
(simple_cnn, resnet50, efficientnet_b0).

Exemplo:
    python src/gradcam.py --image foto.jpg --checkpoint checkpoints/humanos/resnet50 \
        --output gradcam_result.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.cm as cm

from dataset import preprocess_single_image
from face_utils import detect_and_crop_face


def _find_last_conv_layer(container: keras.Model) -> str:
    """Acha a última camada com saída espacial 4D (batch, h, w, canais)
    dentro de um Model/Sequential -- funciona tanto para o simple_cnn quanto
    para o "miolo" de uma backbone do keras.applications."""
    for layer in reversed(container.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("Nenhuma camada com saída espacial encontrada para Grad-CAM.")


def _get_backbone_submodel(model: keras.Model):
    """Nos modelos de transfer learning (resnet50/efficientnet_b0), a
    backbone pré-treinada fica como uma sub-model aninhada dentro do modelo
    principal. Para o simple_cnn não há sub-model (retorna None)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


def _run_layers_eager(model: keras.Model, x: tf.Tensor, stop_before=None, start_after=None) -> tf.Tensor:
    """Roda as camadas de `model` em sequência, de forma totalmente eager
    (sem construir um novo grafo estático). Usado para reproduzir o trecho
    do modelo antes/depois da backbone aninhada, já que no Keras 3 não é
    possível compor um único grafo estático "por dentro" de uma sub-model
    (as saídas intermediárias de uma sub-model pertencem ao grafo interno
    dela, não ao grafo do modelo externo)."""
    skipping = start_after is not None
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if skipping:
            if layer is start_after:
                skipping = False
            continue
        if stop_before is not None and layer is stop_before:
            break
        x = layer(x, training=False)
    return x


def compute_gradcam(model: keras.Model, input_array: np.ndarray, class_idx: int) -> np.ndarray:
    backbone = _get_backbone_submodel(model)
    input_tensor = tf.convert_to_tensor(input_array)

    if backbone is None:
        # simple_cnn: sem sub-model aninhada, dá pra compor um único grafo estático
        target_layer_name = _find_last_conv_layer(model)
        grad_model = keras.Model(model.inputs, [model.get_layer(target_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(input_tensor, training=False)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_output)
        conv_output = conv_output[0]
    else:
        # resnet50/efficientnet_b0: a backbone é uma sub-model aninhada.
        # activation_model e backbone_tail ficam inteiramente DENTRO do
        # grafo interno da própria backbone (por isso não têm o problema de
        # cruzar grafos) -- eles isolam o tensor da última camada conv.
        target_layer_name = _find_last_conv_layer(backbone)
        activation_model = keras.Model(backbone.input, backbone.get_layer(target_layer_name).output)
        backbone_tail = keras.Model(backbone.get_layer(target_layer_name).output, backbone.output)

        head_output = _run_layers_eager(model, input_tensor, stop_before=backbone)

        with tf.GradientTape() as tape:
            conv_output = activation_model(head_output, training=False)
            tape.watch(conv_output)
            backbone_output = backbone_tail(conv_output, training=False)
            predictions = _run_layers_eager(model, backbone_output, start_after=backbone)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_output)
        conv_output = conv_output[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    heatmap = cm.jet(np.array(cam_resized) / 255.0)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = (np.array(image) * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="pasta do checkpoint (contém best_model.keras + metadata.json)")
    parser.add_argument("--output", type=str, default="gradcam_result.png")
    parser.add_argument("--no-face-crop", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    with open(checkpoint_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    classes = metadata["classes"]
    model_name = metadata["model_name"]

    if model_name == "qwen3vl":
        raise SystemExit(
            "Grad-CAM não se aplica ao modelo qwen3vl (ele usa embeddings "
            "pré-extraídos, sem mapa espacial de ativações). Use um "
            "checkpoint treinado com --model simple_cnn/resnet50/efficientnet_b0."
        )

    model = keras.models.load_model(checkpoint_dir / "best_model.keras")
    use_freq_channel = metadata.get("use_freq_channel", metadata.get("in_channels", 3) == 4)

    original_image = Image.open(args.image).convert("RGB")
    display_image = original_image

    if not args.no_face_crop:
        face = detect_and_crop_face(original_image)
        if face is not None:
            display_image = face

    input_array = preprocess_single_image(display_image, use_freq_channel=use_freq_channel)

    logits = model(input_array, training=False)
    pred_idx = int(tf.argmax(logits, axis=1)[0])

    cam = compute_gradcam(model, input_array, pred_idx)

    result = overlay_heatmap(display_image, cam)
    result.save(args.output)

    print(f"Predição: {classes[pred_idx]}")
    print(f"Visualização Grad-CAM salva em: {args.output}")


if __name__ == "__main__":
    main()
