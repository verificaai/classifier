# Classificador de Imagens Reais vs. Geradas por IA

Pipeline completo: geração de dataset sintético (Qwen3-VL + Qwen-Image/FLUX 2)
+ pré-processamento + classificador **TensorFlow/Keras** com transfer
learning sobre embeddings do Qwen3-VL. Estruturado por **categoria** (ex:
"humanos") para você poder adicionar novas categorias no futuro sem
reescrever nada.

## Dois frameworks, cada um onde faz mais sentido

- **Classificador (modelo, treino, avaliação, inferência, Grad-CAM, detecção
  facial)**: TensorFlow/Keras.
- **Extração de embeddings do Qwen3-VL e geração de imagens com
  Qwen-Image/FLUX 2**: PyTorch, via Hugging Face `transformers`/`diffusers`
  — é assim que esses modelos são disponibilizados, não há alternativa em
  TensorFlow para eles.

A ponte entre os dois é o arquivo `.npz` (numpy puro) que
`extract_features.py` gera: ele roda o Qwen3-VL uma vez em PyTorch e salva
o resultado num formato que o restante do pipeline (100% TensorFlow) lê sem
nenhuma dependência de PyTorch.

## Estrutura do projeto

```
ai-image-classifier/
├── README.md
├── requirements.txt
├── src/
│   ├── categories.py           # config por categoria (ADICIONE NOVAS AQUI)
│   ├── face_utils.py           # detecção + recorte facial (MTCNN, TensorFlow)
│   ├── preprocess_faces.py     # pré-processamento (crop facial se a categoria pedir)
│   ├── qwen_features.py        # extrai embedding do Qwen3-VL de uma imagem (PyTorch)
│   ├── extract_features.py     # extrai e cacheia embeddings de todo o dataset (-> .npz)
│   ├── dataset.py              # tf.data: image_dataset_from_directory (CNN) e embeddings cacheados
│   ├── model.py                # Keras: EmbeddingClassifier (Qwen3-VL) + CNNs alternativas
│   ├── train.py                # loop de treino com tf.GradientTape (estilo do projeto de referência)
│   ├── evaluate.py
│   ├── inference.py
│   └── gradcam.py              # explicabilidade (só para os backbones CNN)
├── data_generation/
│   ├── caption_real_images.py  # passo 2: Qwen3-VL gera legendas (PyTorch)
│   └── generate_fake_images.py # passo 3: Qwen-Image/FLUX 2 gera as imagens falsas (PyTorch)
├── data/
│   ├── sample_dataset.py       # amostra um subconjunto de um dataset pronto (ex: CelebA)
│   └── organize_dataset.py     # organiza em train/val/test
└── checkpoints/<categoria>/<modelo>/
    ├── best_model.keras
    ├── metadata.json
    └── training_curves.png
```

## Por que Qwen3-VL como backbone de classificação?

Em vez de treinar uma CNN do zero ou fazer fine-tuning de um ResNet, o
classificador principal (`--model qwen3vl`) usa **embeddings do próprio
Qwen3-VL** (o mesmo modelo já usado para gerar as legendas no passo 2) como
representação da imagem, com uma cabeça de classificação pequena (MLP, em
Keras) por cima. Isso é transfer learning de verdade: aproveita tudo que o
Qwen3-VL já "sabe" sobre imagens sem precisar treinar uma rede do zero.

**Importante sobre custo**: rodar um modelo de bilhões de parâmetros a cada
imagem, a cada época de treino, seria inviável. Por isso o fluxo é em duas
etapas:
1. `extract_features.py` roda o Qwen3-VL **uma única vez** por imagem
   (PyTorch) e salva o embedding em disco como `.npz` (numpy puro).
2. `train.py` treina apenas a cabeça de classificação (poucas camadas
   densas, em Keras) sobre esses embeddings já prontos — rápido e barato,
   mesmo em CPU, sem tocar no Qwen3-VL de novo nem depender de PyTorch.

Os backbones CNN (**ResNet50V2**/**EfficientNetB0**/**SimpleCNN**, via
`keras.applications`) continuam disponíveis como alternativa mais leve (sem
precisar de GPU grande nem baixar um modelo de bilhões de parâmetros) —
úteis para comparação ou se você não tiver hardware para o Qwen3-VL. (O
Keras não tem ResNet18 pronto; ResNet50V2 é o equivalente mais próximo em
`keras.applications`.)

## Extensibilidade: adicionando novas categorias

Tudo é organizado por `--category` (ex: `humanos`). Para adicionar uma nova
categoria no futuro (ex: `paisagens`, `animais`):

1. Abra `src/categories.py` e adicione uma entrada em `CATEGORIES` com o
   prompt de legenda apropriado e se a categoria precisa de recorte facial
   (`apply_face_crop`).
2. Rode o pipeline normalmente passando `--category paisagens` em cada
   script — os caminhos de dados e checkpoints são organizados
   automaticamente por categoria, sem conflitar com `humanos`.

Nenhum outro arquivo precisa ser tocado.

## Pipeline completo, passo a passo (categoria: humanos)

### 1. Fotos reais
Junte fotos reais de pessoas em `raw_data/humanos/real/` (verifique
licenciamento/consentimento se forem rostos de pessoas identificáveis).

Se você já tem um dataset pronto e grande demais para processar por inteiro
(ex: CelebA, com ~200 mil imagens), amostre um subconjunto em vez de apontar
para a pasta inteira — rodar Qwen3-VL + Qwen-Image/FLUX 2 em todas as
imagens seria inviável em tempo/custo:
```bash
python data/sample_dataset.py \
    --source-dir /caminho/para/celeba \
    --output-dir raw_data/humanos/real \
    --n-samples 500
```

### 2. Legendas com Qwen3-VL
Default já ajustado para GPU de 8GB (Qwen3-VL-4B + 4-bit):
```bash
python data_generation/caption_real_images.py --category humanos
```

### 3. Imagens artificiais com Qwen-Image (via API hospedada)
Qwen-Image (20B) não cabe numa GPU de 8-16GB nem quantizado -- por isso este
passo roda via API hospedada em vez de local. Escolha um provedor (veja o
cabeçalho de `generate_fake_images.py` para as instruções de instalação e
variável de ambiente da API key):
```bash
# DashScope (Alibaba Cloud, oficial do Qwen-Image)
python data_generation/generate_fake_images.py --category humanos --api-provider dashscope

# fal.ai (alternativa, cadastro mais simples)
python data_generation/generate_fake_images.py --category humanos --api-provider fal
```

### 4. Pré-processamento (recorte facial, automático para "humanos")
```bash
python src/preprocess_faces.py --category humanos \
    --real-dir raw_data/humanos/real --fake-dir raw_data/humanos/fake
```

### 5. Organizar em train/val/test
```bash
python data/organize_dataset.py --category humanos
```

### 6. Extrair embeddings do Qwen3-VL (uma vez só)
```bash
python src/extract_features.py --data-dir dataset/humanos --output-dir features/humanos
```

### 7. Treinar a cabeça de classificação
```bash
python src/train.py --category humanos --model qwen3vl --features-dir features/humanos --epochs 30
```

### 8. Avaliar
```bash
python src/evaluate.py --checkpoint checkpoints/humanos/qwen3vl \
    --features-dir features/humanos
```

### 9. Classificar uma imagem ou vídeo novo
```bash
# Imagem
python src/inference.py --image foto.jpg --checkpoint checkpoints/humanos/qwen3vl

# Vídeo (extrai 1 frame por segundo, até 30 frames, e agrega o resultado)
python src/inference.py --video clipe.mp4 --checkpoint checkpoints/humanos/qwen3vl

# Vídeo, ajustando amostragem (mais frames = mais preciso, porém mais lento)
python src/inference.py --video clipe.mp4 --checkpoint checkpoints/humanos/qwen3vl \
    --frame-interval 0.5 --max-frames 60
```
Para vídeo, cada frame amostrado é classificado individualmente e o
resultado final é a média das probabilidades entre os frames, junto com o
% de frames que concordam com o veredito (sinal de confiabilidade -- se
os frames discordam muito, o vídeo pode ter trechos mistos ou o resultado
é pouco confiável).

### (Alternativa mais leve, sem Qwen3-VL) — ResNet50V2 com transfer learning
```bash
python src/train.py --category humanos --model resnet50 --data-dir dataset/humanos --epochs 15
python src/gradcam.py --image foto.jpg --checkpoint checkpoints/humanos/resnet50
```

## Decisões de projeto (e por quê)

- **Qwen3-VL como backbone via embeddings cacheados**: transfer learning
  eficiente, evita o custo de rodar um modelo gigante a cada época.
- **Loop de treino com `tf.GradientTape`** (em vez de `model.fit()`): segue
  o padrão do projeto de referência usado como ponto de partida, com early
  stopping por paciência e checkpoint do melhor modelo.
- **Recorte facial (MTCNN, TensorFlow), condicional por categoria**: para
  "humanos", os artefatos de geração por IA se concentram no rosto. A
  config em `categories.py` decide se isso se aplica, então categorias
  futuras que não são de rosto não são afetadas.
- **Canal de resíduo de alta frequência** (disponível para os backbones
  CNN): realça artefatos que GANs/difusão deixam em altas frequências. É
  incorporado à backbone pré-treinada via uma camada "adapter" (Conv2D 1x1)
  inicializada como identidade, para não perturbar os pesos da ImageNet no
  início do treino.
- **Grad-CAM** (backbones CNN): valida que o modelo aprendeu padrões reais
  do rosto/artefatos e não um viés espúrio do dataset.

## Próximos aprimoramentos possíveis

1. **Cross-generator evaluation**: testar se o modelo generaliza para
   imagens de outros geradores (Midjourney, Stable Diffusion) além de
   FLUX 2/Qwen-Image — combinar com uma base pronta (ex: GenImage).
2. **Robustez a compressão/redimensionamento**: aumentar o dataset com
   JPEG compression e resize simulados (fotos reais circulam assim).
3. **Fine-tuning parcial do Qwen3-VL**: em vez de só treinar a cabeça sobre
   embeddings congelados, fazer fine-tuning leve (LoRA) das últimas camadas
   se tiver GPU disponível — pode melhorar a especialização para a tarefa.
4. **Landmarks faciais como sinal extra**: pontos de referência do rosto
   (distância entre olhos, simetria) como feature adicional.
