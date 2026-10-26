from functools import lru_cache

from transformers import MarianMTModel, MarianTokenizer
import torch


DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ar-en"


@lru_cache(maxsize=2)
def _load_translation_model(
    model_name: str,
) -> tuple[MarianTokenizer, MarianMTModel]:
    """Load each translation model once and reuse it across instances."""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


class QueryTranslator:
    """Translate Arabic queries into English for RAG retrieval."""

    def __init__(self, model_name: str = DEFAULT_TRANSLATION_MODEL):
        self.model_name = model_name
        self._tokenizer: MarianTokenizer | None = None
        self._model: MarianMTModel | None = None

    def _ensure_loaded(self) -> None:
        """Defer loading the translation model until it is actually used."""
        if self._model is None:
            self._tokenizer, self._model = _load_translation_model(
                self.model_name
            )

    def translate_to_english(self, text: str) -> str:
        """Translate an Arabic query into English."""
        if not text or not text.strip():
            return text

        self._ensure_loaded()

        inputs = self._tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            translated = self._model.generate(**inputs, max_length=512)

        return self._tokenizer.decode(
            translated[0],
            skip_special_tokens=True,
        ).strip()