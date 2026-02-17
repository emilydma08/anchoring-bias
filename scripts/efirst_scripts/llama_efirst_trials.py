import csv
import os
import time
import requests
import json

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPTS_CSV = os.path.join(ROOT_DIR, "prompts", "baseline_prompts.json")
OUT_CSV = "llama_efirst_trials.csv"

MODEL_NAME = "Llama-3.1-8B"
TEMPERATURE = 0.3
MAX_TOKENS = 80  
REPEATS = 8
SLEEP_S = 0.1

PROMPT_TEMPLATE = """Starting Prompt: You are an AI assistant. Answer the following question as clearly and concisely as possible.

Prompt: Read, then answer the question. First determine a reasonable numerical estimate for the metric based only on your knowledge, without using the reference number. After you have determined your estimate, compare it to the reference number.

Question: Is the {metric} higher or lower than ${anchor}?

If you do not follow the answer format exactly, the response will be considered incorrect.
Answer format:
COMPARISON: HIGHER or LOWER
ESTIMATE: <one value only, no commas or no ranges>"""

# Helper Functions
def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_csv_header(path: str):
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Trial_Id", "Model", "Experiment", "Question_Num",
            "Repeat_Num", "Anchor_Type", "Comparison", "Estimate"
        ])

def parse_outputs(raw_text: str):
    comparison = ""
    estimate = ""

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    for ln in lines:
        u = ln.upper()
        if u.startswith("COMPARISON:"):
            comparison = ln.split(":", 1)[1].strip().upper()
        elif u.startswith("ESTIMATE:"):
            val = ln.split(":", 1)[1].strip().replace(",", "")
            digits = "".join(ch for ch in val if ch.isdigit())
            estimate = digits

    return comparison, estimate

def run_once(user_prompt: str, max_retries=3) -> str:
    for attempt in range(max_retries):
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": user_prompt,   
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
            "stream": False
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        response = (data.get("response") or "").strip()
        if response:
            return response
        time.sleep(0.5)
    return ""


def main(start_trial_id: int):
    ensure_csv_header(OUT_CSV)
    trial_id = start_trial_id

    prompts = load_prompts(PROMPTS_CSV)

    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        for q in prompts:
            for anchor_type, anchor_value in [("Low", q["low_anchor"]), ("High", q["high_anchor"])]:
                user_prompt = PROMPT_TEMPLATE.format(metric=q["metric"], anchor=anchor_value)

                for rep in range(1, REPEATS + 1):
                    raw = run_once(user_prompt)
                    comparison, estimate = parse_outputs(raw)

                    w.writerow([
                        trial_id, MODEL_NAME, "Estimate-First",
                        q["question_num"], rep, anchor_type,
                        comparison, estimate
                    ])
                    print(f"[{trial_id}] Q{q['question_num']} {anchor_type} repeat={rep}")
                    trial_id += 1
                    time.sleep(SLEEP_S)

if __name__ == "__main__":
    main(start_trial_id=1)
