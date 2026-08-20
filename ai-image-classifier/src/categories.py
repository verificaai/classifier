
CATEGORIES = {
    "humanos": {
        # se True, aplica detecção+recorte facial no pré-processamento
        "apply_face_crop": True,
        "caption_prompt": (
            "Describe this photo of a person in detail, focusing on: pose, "
            "facial expression, lighting, framing, clothing, hair and "
            "background. Write a single dense sentence, in the style of an "
            "image generation prompt."
        ),
    },

    # Exemplo de como adicionar uma nova categoria:
    # "paisagens": {
    #     "apply_face_crop": False,
    #     "caption_prompt": (
    #         "Describe this landscape photo in detail: environment type, "
    #         "time of day, weather, dominant colors and composition. Write "
    #         "a single dense sentence, in the style of an image generation "
    #         "prompt."
    #     ),
    # },
}

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail: main subject, colors, composition, "
    "lighting and style. Write a single dense sentence, in the style of an "
    "image generation prompt."
)

DEFAULT_CONFIG = {
    "apply_face_crop": False,
    "caption_prompt": DEFAULT_CAPTION_PROMPT,
}


def get_category_config(category: str) -> dict:
    """Retorna a config da categoria, ou uma config genérica se a categoria
    ainda não tiver sido cadastrada em CATEGORIES."""
    return CATEGORIES.get(category, DEFAULT_CONFIG)


def list_categories():
    return sorted(CATEGORIES.keys())
