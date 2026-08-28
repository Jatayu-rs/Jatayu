import torch
from typing import Protocol
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Translator(Protocol):
    def to_english(self, text: str, src_lang: str) -> str: ...
    def from_english(self, text: str, tgt_lang: str) -> str: ...


class NullTranslator:
   
    def to_english(self, text: str, src_lang: str) -> str:
        return text

    def from_english(self, text: str, tgt_lang: str) -> str:
        return text


class IndicTrans2Translator:
    
    _INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
    _EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"

    def __init__(self, hf_token: str):
        self.hf_token = "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"
        self._cache = {}

    def _load(self, model_id: str):
        if model_id in self._cache:
            return self._cache[model_id]
            
        try:
            from IndicTransToolkit.processor import IndicProcessor  # type: ignore
        except ImportError:
            print("[Dependency Critical Warning] IndicTransToolkit not found.")
            raise


        print(f"[NLP Core] Initializing parameters for resource model target: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=self.hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True, token=self.hf_token)
        ip = IndicProcessor(inference=True)
        
        self._cache[model_id] = (tokenizer, model, ip)
        return self._cache[model_id]

    def to_english(self, text: str, src_lang: str) -> str:
        if src_lang == "eng_Latn":
            return text
        try:
            return self._run(self._INDIC_EN, text, src_lang, "eng_Latn")
        except Exception as e:
            print(f"[Translation Failed Exception] to_english fallback initialized: {e}")
            return text  

    def from_english(self, text: str, tgt_lang: str) -> str:
        if tgt_lang == "eng_Latn":
            return text
        try:
            return self._run(self._EN_INDIC, text, "eng_Latn", tgt_lang)
        except Exception as e:
            print(f"[Translation Failed Exception] from_english fallback initialized: {e}")
            return text

    def _run(self, model_id: str, text: str, src: str, tgt: str) -> str:
        tok, model, ip = self._load(model_id)
        batch = ip.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
        inputs = tok(batch, truncation=True, padding=True, return_tensors="pt")
        
        with torch.inference_mode():
            out = model.generate(**inputs, num_beams=5, max_length=256)
            
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        return ip.postprocess_batch(decoded, lang=tgt)[0]
