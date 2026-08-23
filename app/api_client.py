import requests

BASE_URL = "http://localhost:8501"
MOCK_MODE = True  # flip to False once the backend exists


def query(files, question, lang="en"):
    if MOCK_MODE:
        return _mock_response(lang)

    resp = requests.post(
        f"{BASE_URL}/query",
        files=files,
        data={"query": question, "lang": lang},
    )
    return resp.json()


def _mock_response(lang):
    return {
        "answer": "There are approximately 340 buildings visible in this scene.",
        "answer_translated": "इस दृश्य में लगभग 340 इमारतें दिखाई दे रही हैं।",
        "answer_english": "There are approximately 340 buildings visible in this scene.",
        "confidence": 0.42,
        "trace": [
            {"stage": "check", "label": "Validated inputs", "duration_ms": 120},
            {"stage": "task", "label": "Parsed question intent", "duration_ms": 80},
            {"stage": "select", "label": "Selected VQA specialist", "duration_ms": 40},
            {"stage": "run", "label": "Ran model inference", "duration_ms": 2300},
            {"stage": "assemble", "label": "Assembled evidence + report", "duration_ms": 60},
        ],
        "legend": {
            "Water": "#3B82F6",
            "Buildings": "#EF4444",
            "Vegetation": "#10B981",
            "Bare soil": "#F59E0B",
        },
        "overlay_png": None,  # swap in a real placeholder path once you have one
        "issues": [],
    }