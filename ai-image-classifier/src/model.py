"""
Modelos em TensorFlow/Keras (Functional API), seguindo o padrão do projeto
de referência (Unfake), que também é implementado em TF/Keras.

Três caminhos de modelo:

1. build_simple_cnn: baseline rápido do zero (4 blocos conv+bn+relu+maxpool
   + dense), adaptado da CNN do projeto de referência.

2. build_transfer_model: backbone pré-treinada na ImageNet (ResNet50V2 ou
   EfficientNetB0) via keras.applications, com a cabeça de classificação
   trocada. O Keras não tem ResNet18 pronto -- ResNet50V2 é o equivalente
   mais próximo disponível em keras.applications.

   Suporta `in_channels=4` (canal extra de resíduo de alta frequência, ver
   src/dataset.py) através de uma camada "adapter": um Conv2D 1x1 que mistura
   os `in_channels` de entrada para os 3 canais que o backbone pré-treinado
   espera. Ela é inicializada como uma identidade nos 3 primeiros canais e
   peso zero no(s) canal(is) extra(s) -- ou seja, no início do treino o
   modelo se comporta exatamente como se tivesse recebido só RGB, e aprende
   a incorporar o sinal extra durante o treino.

3. build_embedding_classifier: cabeça de classificação leve (MLP) treinada
   sobre embeddings pré-extraídos do Qwen3-VL (ver src/qwen_features.py e
   src/extract_features.py) -- a forma eficiente de fazer transfer learning
   com um modelo gigante sem precisar rodá-lo a cada época.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_simple_cnn(num_classes: int = 2, in_channels: int = 3, img_size: int = 224) -> keras.Model:
    inputs = keras.Input(shape=(img_size, img_size, in_channels))
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    for filters in (16, 32, 64, 64):
        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)

    return keras.Model(inputs, outputs, name="simple_cnn")


def _channel_adapter(inputs, in_channels: int):
    """Conv2D 1x1 que mistura in_channels -> 3, inicializado como identidade
    nos 3 primeiros canais (RGB) e zero no(s) extra(s)."""
    if in_channels == 3:
        return inputs

    identity_init = np.zeros((1, 1, in_channels, 3), dtype=np.float32)
    for c in range(3):
        identity_init[0, 0, c, c] = 1.0

    adapter = keras.layers.Conv2D(
        3, kernel_size=1, use_bias=False, name="channel_adapter",
        kernel_initializer=keras.initializers.Constant(identity_init),
    )
    return adapter(inputs)


def build_transfer_model(model_name: str, num_classes: int = 2, in_channels: int = 3,
                          img_size: int = 224, freeze_backbone: bool = False) -> keras.Model:
    if model_name == "resnet50":
        base = keras.applications.ResNet50V2(
            include_top=False, weights="imagenet",
            input_shape=(img_size, img_size, 3), pooling="avg")
        # ResNetV2 espera entrada em [-1, 1]
        normalize = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescale_resnet")
    elif model_name == "efficientnet_b0":
        base = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(img_size, img_size, 3), pooling="avg")
        # EfficientNet já normaliza internamente -- espera [0, 255] cru
        normalize = keras.layers.Activation("linear", name="identity_efficientnet")
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    base.trainable = not freeze_backbone

    inputs = keras.Input(shape=(img_size, img_size, in_channels))
    x = _channel_adapter(inputs, in_channels)
    x = normalize(x)
    # training=False mantém as estatísticas de BatchNorm congeladas nos
    # valores da ImageNet, mesmo com o backbone treinável -- prática
    # recomendada para fine-tuning com datasets pequenos (evita que poucas
    # centenas de imagens desestabilizem as médias/variâncias do BatchNorm).
    x = base(x, training=False)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)

    return keras.Model(inputs, outputs, name=f"{model_name}_classifier")


def build_embedding_classifier(embedding_dim: int, num_classes: int = 2,
                                hidden_dim: int = 256, dropout: float = 0.3) -> keras.Model:
    inputs = keras.Input(shape=(embedding_dim,))
    x = keras.layers.Dense(hidden_dim, activation="relu")(inputs)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)

    return keras.Model(inputs, outputs, name="embedding_classifier")


def build_model(model_name: str, num_classes: int = 2, in_channels: int = 3,
                 embedding_dim: int | None = None, img_size: int = 224,
                 freeze_backbone: bool = False) -> keras.Model:
    if model_name == "simple_cnn":
        return build_simple_cnn(num_classes=num_classes, in_channels=in_channels, img_size=img_size)
    if model_name == "qwen3vl":
        assert embedding_dim is not None, "embedding_dim é obrigatório para o modelo qwen3vl"
        return build_embedding_classifier(embedding_dim=embedding_dim, num_classes=num_classes)
    return build_transfer_model(model_name, num_classes=num_classes, in_channels=in_channels,
                                 img_size=img_size, freeze_backbone=freeze_backbone)
