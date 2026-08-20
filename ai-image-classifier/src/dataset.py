"""
Carregamento de dados em TensorFlow (tf.data), com dois caminhos:

build_image_dataset: para os modelos baseados em imagem (simple_cnn,
   resnet50, efficientnet_b0). Usa `image_dataset_from_directory`, com
   augmentation (flip/rotação/contraste) no treino e um canal opcional de
   "resíduo de alta frequência" concatenado ao RGB.

load_feature_dataset: para o modelo qwen3vl, que treina sobre embeddings
   já extraídos e cacheados por src/extract_features.py (arquivos .npz).

Por que o canal de frequência: GANs e modelos de difusão tendem a deixar
padrões sutis de textura/ruído em altas frequências (bordas, cabelo, pele)
que não são tão óbvios no RGB puro. O canal extra é: grayscale -
blur(grayscale), um filtro passa-alta simples que realça esses artefatos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

IMG_SIZE = 224  # padrão para ResNet/EfficientNet pré-treinados na ImageNet

_GAUSSIAN_KERNEL_NP = None


def _get_gaussian_kernel(size: int = 5, sigma: float = 2.0) -> tf.Tensor:
    # Cacheamos só o numpy array. Um tf.constant criado dentro de um
    # tf.data.Dataset.map() fica preso ao FuncGraph daquele map e não pode
    # ser reaproveitado fora dele (erro "out of scope") -- por isso o
    # tf.constant é recriado a cada chamada a partir do array numpy, que é
    # barato e funciona tanto em modo eager quanto dentro de tf.function.
    global _GAUSSIAN_KERNEL_NP
    if _GAUSSIAN_KERNEL_NP is None:
        ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        _GAUSSIAN_KERNEL_NP = kernel.reshape(size, size, 1, 1).astype(np.float32)
    return tf.constant(_GAUSSIAN_KERNEL_NP)


def _gaussian_blur(gray: tf.Tensor) -> tf.Tensor:
    """gray: [H, W, 1] -> [H, W, 1] borrado."""
    kernel = _get_gaussian_kernel()
    x = gray[tf.newaxis, ...]
    blurred = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return blurred[0]


def compute_freq_channel(image: tf.Tensor) -> tf.Tensor:
    """image: [H, W, 3] float32 em [0, 255] -> canal de resíduo [H, W, 1]."""
    gray = tf.image.rgb_to_grayscale(image)
    blurred = _gaussian_blur(gray)
    residual = gray - blurred
    std = tf.math.reduce_std(residual) + 1e-6
    return residual / std


def _build_augmentation() -> keras.Sequential:
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(10 / 360),
        keras.layers.RandomContrast(0.1),
    ], name="augmentation")


def build_image_dataset(data_dir: str, split: str, batch_size: int = 32,
                         use_freq_channel: bool = True, img_size: int = IMG_SIZE,
                         seed: int = 42):
    """Retorna (tf.data.Dataset, class_names) para train/val/test de
    data_dir/split/{real,fake}/*.jpg (estrutura do torchvision ImageFolder,
    também aceita nativamente pelo Keras)."""
    directory = Path(data_dir) / split
    train = split == "train"

    raw_ds = keras.utils.image_dataset_from_directory(
        directory, labels="inferred", label_mode="int",
        image_size=(img_size, img_size), batch_size=None,
        shuffle=train, seed=seed,
    )
    class_names = raw_ds.class_names  # ordenado alfabeticamente: ['fake', 'real']

    augment = _build_augmentation() if train else None

    def _map(image, label):
        image = tf.cast(image, tf.float32)
        if augment is not None:
            image = augment(image, training=True)
        if use_freq_channel:
            freq = compute_freq_channel(image)
            image = tf.concat([image, freq], axis=-1)
        return image, tf.cast(label, tf.int32)

    ds = raw_ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.shuffle(1000, seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, class_names


def preprocess_single_image(image: Image.Image, use_freq_channel: bool = True,
                             img_size: int = IMG_SIZE) -> np.ndarray:
    """Prepara uma única imagem PIL para inferência/Grad-CAM, retornando um
    batch de tamanho 1: [1, H, W, 3 ou 4]. Não aplica a normalização
    específica do backbone (ResNet/EfficientNet/etc) -- isso já está embutido
    como camada dentro do modelo salvo (ver src/model.py)."""
    image = image.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    array = tf.cast(np.array(image), tf.float32)

    if use_freq_channel:
        freq = compute_freq_channel(array)
        array = tf.concat([array, freq], axis=-1)

    return array.numpy()[np.newaxis, ...]


class FeatureDataset:
    """Carrega embeddings pré-extraídos (ex: pelo Qwen3-VL), salvos por
    src/extract_features.py em um .npz com arrays "embeddings", "labels" e
    "classes"."""

    def __init__(self, features_path):
        data = np.load(features_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.labels = data["labels"].astype(np.int32)
        self.classes = [str(c) for c in data["classes"]]
        # Config do Qwen3-VL usada para gerar estes embeddings (ausente em
        # .npz antigos, gerados antes desses campos existirem -- por isso o
        # default aqui precisa bater com o default de extract_embedding em
        # src/qwen_features.py).
        self.qwen_model_id = (
            str(data["qwen_model_id"]) if "qwen_model_id" in data else "Qwen/Qwen3-VL-4B-Instruct"
        )
        self.qwen_use_4bit = (
            bool(data["qwen_use_4bit"]) if "qwen_use_4bit" in data else True
        )

    @property
    def embedding_dim(self):
        return self.embeddings.shape[1]

    def __len__(self):
        return len(self.labels)


def load_feature_dataset(features_dir: str, split: str, batch_size: int = 32):
    path = Path(features_dir) / f"{split}.npz"
    fd = FeatureDataset(path)

    ds = tf.data.Dataset.from_tensor_slices((fd.embeddings, fd.labels))
    if split == "train":
        ds = ds.shuffle(max(len(fd), 1), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    qwen_info = {"qwen_model_id": fd.qwen_model_id, "qwen_use_4bit": fd.qwen_use_4bit}
    return ds, fd.classes, fd.embedding_dim, qwen_info
