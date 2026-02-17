import csv
import os
import time
import requests
import json
from ollama import Client

# Configuration
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
MISTRAL_MODEL = "mistral-large-3:675b-cloud"

client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPTS_CSV = os.path.join(ROOT_DIR, "prompts", "baseline_prompts.json")
OUT_CSV = "mistral_debias_trials.csv"

MODEL_NAME = "Mistral-Large-3"
TEMPERATURE = 0.3
MAX_TOKENS = 80  
REPEATS = 8
SLEEP_S = 0.1

PROMPT_TEMPLATE = """Starting Prompt: You are an AI assistant. Answer the following question as clearly and concisely as possible.

Prompt: Read, then answer the question. The value provided in the question is an irrelevant reference number and may bias the estimate. You should generate your numerical estimate independently of this value.

Question: Is the {metric} higher or lower than ${anchor}?

Please answer with only: HIGHER or LOWER.
Then estimate the value in dollars.

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

    if not comparison:
        for ln in lines:
            if ln.strip().upper() in ("HIGHER", "LOWER"):
                comparison = ln.strip().upper()
                break
    if not estimate:
        for ln in lines:
            stripped = ln.strip().upper()
            if stripped in ("HIGHER", "LOWER") or stripped.startswith("COMPARISON:"):
                continue
            val = ln.strip().replace(",", "").replace("$", "")
            digits = "".join(ch for ch in val if ch.isdigit())
            if digits:
                estimate = digits
                break

    return comparison, estimate

def run_once_cloud(user_prompt: str, max_retries=3) -> str:
    messages = [{'role': 'user', 'content': user_prompt}]

    for attempt in range(max_retries):
        try:
            response = client.chat(
                model="mistral-large-3:675b-cloud",
                messages=messages,
                stream=False,
            )
            raw_text = response.message.content.strip()
            if raw_text:
                return raw_text

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
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
                    raw = run_once_cloud(user_prompt)
                    comparison, estimate = parse_outputs(raw)

                    w.writerow([
                        trial_id, MODEL_NAME, "Debias",
                        q["question_num"], rep, anchor_type,
                        comparison, estimate
                    ])
                    print(f"[{trial_id}] Q{q['question_num']} {anchor_type} repeat={rep}")
                    trial_id += 1
                    time.sleep(SLEEP_S)

if __name__ == "__main__":
    main(start_trial_id=401)
