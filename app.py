import os
import torch
import gradio as gr
from PIL import Image, ImageOps
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from bhashini_nlp import BhashiniTranslator  # Imports your specific NLP code block

# =====================================================================
# 1. INITIALIZE GLOBAL ENGINES & MODEL STRUCTURE
# =====================================================================
# Instantiate your Bhashini Translator Module
BHASHINI_API_KEY = "YOUR_BHASHINI_API_KEY" 
translator = BhashiniTranslator(api_key=BHASHINI_API_KEY)

print("Initializing SatQuery AI Pipeline Engine... (Loading weights)")
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
print(f"🔥 SatQuery Vision Engine loaded successfully on device: {device}")


# =====================================================================
# 2. CORE DUAL-IMAGE PROCESSING & VISION LOGIC
# =====================================================================
def run_vision_inference(image_1: Image, image_2: Image, english_query_text: str):
    """
    Feeds both before/after images and the translated English query to Qwen2.5-VL.
    Generates a text analysis and constructs a localized colored heat map overlay.
    """
    # Create the change analysis layout structure for Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_1},
                {"type": "image", "image": image_2},
                {
                    "type": "text", 
                    "text": (
                        f"System: You are SatQuery AI, an expert Remote Sensing analyst. "
                        f"Compare these two temporal overhead satellite images (Image 1 is Before, Image 2 is After). "
                        f"Provide a structured analysis answering the user's specific request.\n"
                        f"User Request: {english_query_text}"
                    )
                }
            ]
        }
    ]
    
    # Process tensors through the Vision Encoder pipeline
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_1, image_2], padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=300)
        
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    english_explanation_output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    # --- GEOSPATIAL MAP SEGMENTATION OVERLAY GENERATION ---
    # Convert images to grayscale arrays to isolate changing pixels mathematically
    gray1 = ImageOps.grayscale(image_1)
    gray2 = ImageOps.grayscale(image_2)
    
    # Create a dynamic mask matrix showing structural drift over time
    from PIL import ImageChops
    diff_mask = ImageChops.difference(gray1, gray2)
    diff_mask = ImageChops.multiply(diff_mask, ImageOps.colorize(diff_mask, (0,0,0), (0, 230, 115))) # Emerald Tint
    
    colored_map_output = Image.blend(image_2.convert("RGB"), diff_mask.convert("RGB"), alpha=0.55)
    
    return english_explanation_output, colored_map_output


# =====================================================================
# 3. INTERFACING AND TRANSLATION WRAPPER
# =====================================================================
def satquery_orchestrator(image_1, image_2, raw_user_text, user_language):
    """
    The orchestrator that intercepts inputs, manages translations, executes 
    vision steps, and converts results back to the source dialect.
    """
    if image_1 is None or image_2 is None:
        return "⚠️ Error: Please upload both temporal satellite images.", None
    if not raw_user_text.strip():
        return "⚠️ Error: Question box cannot be blank.", None
        
    # Map selection boxes to native language codes
    language_mapping = {"Hindi": "hi", "Bengali": "bn"}
    source_lang_code = language_mapping.get(user_language, "hi")
    
    print(f"\n[NLP Input] Parsing user query in {user_language}: {raw_user_text}")
    
    # STEP A: Local Language -> English Core Translation
    english_query = translator.translate(raw_user_text, source_lang=source_lang_code, target_lang="en")
    print(f"[NLP Status] Dispatched English Query to Vision Core: '{english_query}'")
    
    # STEP B: Run Qwen Engine Processing
    print("[Vision Status] Running dual-image scene analysis inference...")
    english_analysis, highlighted_map = run_vision_inference(image_1, image_2, english_query)
    
    # STEP C: English Output -> Regional Dialect Re-localization
    print(f"[NLP Status] Re-localizing analytical metrics back to {user_language}...")
    native_explanation = translator.translate(english_analysis, source_lang="en", target_lang=source_lang_code)
    
    print("[System Status] Data packages successfully assembled. Rendering UI elements.")
    return native_explanation, highlighted_map


# =====================================================================
# 4. DESIGNING THE VISUAL GRADIO VIEWPORTS
# =====================================================================
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # 🛰️ SatQuery AI Dashboard
        ### Multimodal Remote Sensing Assistant with Native Language Orchestration
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Media Controls")
            sat_t1 = gr.Image(type="pil", label="Satellite Image T1 (Before / Baseline)")
            sat_t2 = gr.Image(type="pil", label="Satellite Image T2 (After / Latest)")
            
            lang_selector = gr.Dropdown(choices=["Hindi", "Bengali"], value="Hindi", label="🌐 Script Dialect")
            query_box = gr.Textbox(lines=3, placeholder="Ask a question about geographic features...", label="💬 Regional Question")
            submit_btn = gr.Button("🚀 Run Multimodal AI Execution", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Spatial Analytics Engine Output")
            map_viewport = gr.Image(type="pil", label="🎨 Analytics Color-Map Overlays")
            text_viewport = gr.Textbox(lines=8, label="📝 Regional Metrics Explanation")

    submit_btn.click(
        fn=satquery_orchestrator,
        inputs=[sat_t1, sat_t2, query_box, lang_selector],
        outputs=[text_viewport, map_viewport]
    )

if __name__ == "__main__":
    # share=True provides an immediate access link you can copy-paste to your teammates!
    demo.launch(share=True)
