import torch

from config import Config
from gpu_manager import GPUManager
from models import model_cache


class TranslationModelWrapper:
    def __init__(self, config: Config):
        self.config = config
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        GPUManager.acquire("translate")
        key = ("translation", self.config.TRANSLATION_MODEL, self.config.TRANSLATION_DEVICE)
        cached = model_cache.get(key)
        if cached is not None:
            self._tokenizer, self._model = cached
        else:
            from transformers import MarianMTModel, MarianTokenizer
            self._tokenizer = MarianTokenizer.from_pretrained(self.config.TRANSLATION_MODEL)
            self._model = MarianMTModel.from_pretrained(self.config.TRANSLATION_MODEL)
            if self.config.TRANSLATION_DEVICE == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            model_cache.put(key, (self._tokenizer, self._model))
        GPUManager.set_model(self._model)

    def translate(self, texts: list[str]) -> list[str]:
        """Translate a list of texts from EN to PT. Returns translated texts."""
        if not texts:
            return []
        # The en-pt model serves both variants; without a target token it drifts
        # into European Portuguese ("estás a entregar"). >>pob<< = pt-BR.
        texts = [f"{self.config.OPUS_TARGET_TOKEN} {t}" for t in texts]
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        if next(self._model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            translated = self._model.generate(**inputs, max_length=512)
        return [self._tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def translate_single(self, text: str) -> str:
        return self.translate([text])[0]

    def unload(self) -> None:
        # Cached in model_cache — stays loaded for the next job.
        self._model = None
        self._tokenizer = None
        GPUManager.release("translate")