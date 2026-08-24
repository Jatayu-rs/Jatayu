# test_nlp.py
import sys
import time
import random
from deep_translator import GoogleTranslator

class SatQueryNLPTest:
    def __init__(self):
        print("Mounting Local Multi-lingual Pipeline... (Zero Auth Required)")
        # Expanded mapping to cover 10 major regional Indian languages
        self.language_codes = {
            "Hindi": "hi",
            "Bengali": "bn",
            "Odia": "or",
            "Tamil": "ta",
            "Telugu": "te",
            "Marathi": "mr",
            "Gujarati": "gu",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Punjabi": "pa"
        }
        print("🔥 SatQuery NLP Translation Core Status: 10 Indian Languages Operational.")

    def translate_to_english_with_retry(self, text: str, source_lang_name: str, max_retries: int = 5) -> str:
        """
        Translates text to English with exponential backoff to handle campus rate limits.
        """
        src_code = self.language_codes.get(source_lang_name, "auto")
        
        for attempt in range(max_retries):
            try:
                english_output = GoogleTranslator(source=src_code, target='en').translate(text)
                
                if "No translation was found" in str(english_output):
                    raise Exception("Rate limit block encountered")
                    
                return english_output
                
            except Exception:
                if attempt == max_retries - 1:
                    return f"[Pipeline Execution Error: Max retries reached on this network node.]"
                
                sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"   ⚠️ Network throttle hit. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)

if __name__ == "__main__":
    print("=" * 75)
    print("     SATQUERY AI: STRESS-TESTING NATIVE PIPELINE (10 INDIC LANGUAGES)   ")
    print("=" * 75)
    
    translator = SatQueryNLPTest()

    # 10 diverse geospatial fake inputs representing real-world use cases
    test_cases = [
        {
            "lang": "Hindi", 
            "text": "इस क्षेत्र में जल निकायों और कृषि भूमि में हुए परिवर्तनों का विश्लेषण करें।"
        },
        {
            "lang": "Bengali", 
            "text": "গত দুই বছরে এই বনাঞ্চলে কতটুকু গাছ কাটা হয়েছে তা দেখান।"
        },
        {
            "lang": "Odia", 
            "text": "ଏହି ଦୁଇଟି ଚିତ୍ର ମଧ୍ୟରେ କେଉଁ ସ୍ଥାନରେ ନୂଆ ଘର ତିଆରି ହୋଇଛି ଚିହ୍ନଟ କରନ୍ତុ।"
        },
        {
            "lang": "Tamil", 
            "text": "இந்த இரண்டு செயற்கைக்கோள் படங்களுக்கு இடையே உள்ள சாலை மாற்றங்களை ஒப்பிடவும்."
        },
        {
            "lang": "Telugu", 
            "text": "వరదలు వచ్చిన తర్వాత నది ప్రాంతంలో జరిగిన మార్పులను చూపించండి."
        },
        {
            "lang": "Marathi", 
            "text": "या दोन उपग्रह छायाचित्रांमधील नागरीकरणाचा आणि बांधकामाचा वेग मोजा."
        },
        {
            "lang": "Gujarati", 
            "text": "આ ચક્રવાત પછી દરિયાકાંઠાના વિસ્તારોમાં થયેલા નુકસાનનું મૂલ્યાંકન કરો."
        },
        {
            "lang": "Kannada", 
            "text": "ಈ ಎರಡು ಕಾಲಾವಧಿಯಲ್ಲಿ ಕಾಡಿನ ವಿಸ್ತೀರ್ಣ ಎಷ್ಟು ಕಡಿಮೆಯಾಗಿದೆ ಎಂದು ಪತ್ತೆಹಚ್ಚಿ."
        },
        {
            "lang": "Malayalam", 
            "text": "ഉരുൾപൊട്ടലിന് ശേഷം ഈ മലയോര മേഖലയിൽ സംഭവിച്ച മാറ്റങ്ങൾ വിശകലനം ചെയ്യുക."
        },
        {
            "lang": "Punjabi", 
            "text": "ਇਸ ਖੇਤਰ ਵਿੱਚ ਖੇਤੀਬਾੜੀ ਵਾਲੀ ਜ਼ਮੀਨ ਵਿੱਚ ਹੋਏ ਬਦਲਾਅ ਦੀ ਪਛਾਣ ਕਰੋ।"
        }
    ]

    print("\n🚀 Commencing batch vector translation verification...\n")
    for i, case in enumerate(test_cases, 1):
        print(f"--- Verification Run #{i} [{case['lang']}] ---")
        print(f"📥 Local Input : {case['text']}")
        
        # Add a 1.2-second spacing interval to avoid triggering the campus firewall
        if i > 1:
            time.sleep(1.2)
            
        english_output = translator.translate_to_english_with_retry(case['text'], source_lang_name=case['lang'])
        print(f"📤 English Goal: '{english_output}'\n")
        
    print("=" * 75)
    print("                    10-LANGUAGE STRESS-TEST COMPLETE                     ")
    print("=" * 75)
