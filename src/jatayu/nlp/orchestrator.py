from src.jatayu.nlp.script_detect import detect_language
from src.jatayu.nlp.translate import IndicTrans2Translator, NullTranslator

try:
    from src.jatayu.analysis.ontology import ALIAS_TABLE  
except ImportError:
    ALIAS_TABLE = {"ory_Orya": {"ଜଳାଶୟ": "water body"}, "ben_Beng": {"জলাশয়": "water body"}}

class JatayuNLPPipeline:
    def __init__(self, hf_token: str):
        try:
            self.translator = IndicTrans2Translator(hf_token="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")
        except Exception:
            self.translator = NullTranslator()

    def process_incoming_query(self, raw_query: str) -> tuple[str, str]:
       
        # Stage 1: Script Detection
        lang_code = detect_language(raw_query)
        print(f"[Pipeline API Boundary] Detected Language FLORES code: {lang_code}")
        
        if lang_code == "eng_Latn":
            return raw_query, lang_code

        # Stage 2: Ontology Alias Lookup on original regional text
        modified_query = raw_query
        if lang_code in ALIAS_TABLE:
            for native_term, english_alias in ALIAS_TABLE[lang_code].items():
                if native_term in modified_query:
                    modified_query = modified_query.replace(native_term, english_alias)
                    print(f"[Stage 2 Noun Catch] Swapped out '{native_term}' for '{english_alias}'")

        # Stage 3: Translate connective tissue structure safely down to English
        english_query = self.translator.to_english(modified_query, src_lang=lang_code)
        return english_query, lang_code

    def process_outgoing_answer(self, english_answer: str, target_lang_code: str) -> str:
        """Stage 5: Localizes the English orchestrator output back to user dialect."""
        return self.translator.from_english(english_answer, tgt_lang=target_lang_code)
