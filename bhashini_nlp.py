import requests
import json

class BhashiniTranslator:
    def __init__(self, api_key: str):
        """
        Initializes the Bhashini Translation Engine.
        Get your free API Key from the Bhashini API Portal (bhashini.gov.in)
        """
        self.api_key = api_key
        self.base_url = "https://bhashini.gov.in"
        
        # Core configuration required by Bhashini's ULCA architecture
        self.master_config_url = "https://bhashini.gov.in"

    def _get_pipeline_config(self, source_lang: str, target_lang: str):
        """Internal helper to negotiate endpoints with Bhashini's gateway."""
        headers = {
            "userID": "satquery_ai_sih2026", # Unique team identifier
            "ulcaApiKey": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_lang,
                            "targetLanguage": target_lang
                        }
                    }
                }
            ],
            "pipelineRequestConfig": {"pipelineId": "65132dd72723232ba8154e3d"} # Standard Translation Pipeline ID
        }
        
        response = requests.post(self.master_config_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Bhashini Authentication Failed: {response.text}")
            
        config_data = response.json()
        
        # Extract the compute endpoints dynamically assigned for today
        service_id = config_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
        callback_url = config_data["pipelineResponseConfig"][0]["config"][0]["inferenceApiKey"]["value"]
        
        return service_id, callback_url

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text across Indic languages.
        Language codes: 'hi' (Hindi), 'bn' (Bengali), 'en' (English)
        """
        try:
            # 1. Fetch live routing configurations
            service_id, inference_token = self._get_pipeline_config(source_lang, target_lang)
            
            # 2. Build the structural translation payload
            headers = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://thunderclient.com)",
                "Authorization": inference_token,
                "Content-Type": "application/json"
            }
            
            payload = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "language": {
                                "sourceLanguage": source_lang,
                                "targetLanguage": target_lang
                            },
                            "serviceId": service_id
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": text}]
                }
            }
            
            # 3. Fire the request to the compute node
            response = requests.post(self.base_url, headers=headers, json=payload)
            if response.status_code == 200:
                translated_text = response.json()["pipelineResponse"][0]["output"][0]["target"]
                return translated_text
            else:
                print(f"[Bhashini Warning] Compute node failed, fallback initiated. Error: {response.text}")
                return self._fallback_translator(text, source_lang, target_lang)
                
        except Exception as e:
            print(f"[NLP System Alert] Dynamic routing failed ({e}). Executing secure fallback...")
            return self._fallback_translator(text, source_lang, target_lang)

    def _fallback_translator(self, text: str, source_lang: str, target_lang: str) -> str:
        """Ensures your team's UI demo never crashes on stage if the gov network lags."""
        # Simple, zero-auth backup using a public mirror engine
        try:
            url = f"https://googleapis.com{source_lang}&tl={target_lang}&dt=t&q={text}"
            res = requests.get(url).json()
            return res[0][0][0]
        except:
            return f"[Translation Error] Could not process query '{text}'"
