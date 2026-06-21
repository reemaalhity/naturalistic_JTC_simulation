from __future__ import annotations

import json
import itertools
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from openai import OpenAI

# =====================================================
# CONFIG
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SEQUENCES_DIR = PROJECT_ROOT / "sequences"
RUNS_DIR = PROJECT_ROOT / "test_run_naturalistic"

REASSURANCE_INSERT_AFTER_INDEX = 0   # after excerpt 2
MAX_EXCERPTS = 6
N_REPEATS = 10

AGENT_TYPES = ["jtc", "nonjtc"]
REASSURANCE_TYPES = ["calibrated", "miscalibrated"]

MODELS = ["openai/gpt-5.4"]

TEMPERATURE = 0.7
REASSURANCE_TEMPERATURE = 0.7

MAX_TOKENS = 120
REASSURANCE_MAX_TOKENS = 40

API_SLEEP = 0.8

# =====================================================
# CLIENT
# =====================================================

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "Missing OPENROUTER_API_KEY.\n"
        'Run in terminal:\nexport OPENROUTER_API_KEY="your_key_here"'
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# =====================================================
# DATA CLASSES
# =====================================================

@dataclass
class Condition:
    agent_type: str
    reassurance_type: str
    sequence_file: str

    @property
    def name(self) -> str:
        seq = Path(self.sequence_file).stem
        return f"{self.agent_type}__{self.reassurance_type}__{seq}"


@dataclass
class TurnRecord:
    turn_index: int
    excerpt: str
    reassurance_shown: bool
    reassurance_text: str | None
    raw_response: str
    parsed_guess: str | None
    parsed_confidence: int | None
    parsed_action: str | None


@dataclass
class RunResult:
    timestamp: str
    model_name: str
    condition: Dict[str, Any]
    sequence_name: str
    true_author: str
    final_guess: str | None
    final_confidence: int | None
    committed: bool
    toc_turn: int | None
    accuracy: int | None
    evidence_requests: int
    turns: List[Dict[str, Any]]

# =====================================================
# HELPERS
# =====================================================

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def validate_files() -> None:
    required = [
        "base_task.txt",
        "jtc.txt",
        "nonjtc.txt",
        "calibrated.txt",
        "miscalibrated.txt",
    ]

    for fname in required:
        path = PROMPTS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing prompt file: {path}")

    if not list(SEQUENCES_DIR.glob("*.json")):
        raise FileNotFoundError(f"No sequence JSON files found in {SEQUENCES_DIR}")

# =====================================================
# PROMPTS
# =====================================================

def build_system_prompt(agent_type: str) -> str:
    base = load_text(PROMPTS_DIR / "base_task.txt")
    style = load_text(PROMPTS_DIR / f"{agent_type}.txt")
    return f"{base}\n\n{style}"


def load_reassurance_prompt(reassurance_type: str) -> str:
    return load_text(PROMPTS_DIR / f"{reassurance_type}.txt")

# =====================================================
# PARSER
# =====================================================

def normalize_label(x: str | None) -> str | None:
    if x is None:
        return None
    return x.strip().lower().rstrip(".")


def parse_response(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    out = {
        "guess": None,
        "confidence": None,
        "reasoning": None,
        "action": None,
    }

    for line in lines:
        low = line.lower()

        if low.startswith("current best guess:"):
            out["guess"] = line.split(":", 1)[1].strip()

        elif low.startswith("confidence:"):
            val = line.split(":", 1)[1].strip()
            digits = "".join(ch for ch in val if ch.isdigit())
            out["confidence"] = int(digits) if digits else None

        elif low.startswith("reasoning:"):
            out["reasoning"] = line.split(":", 1)[1].strip()

        elif low.startswith("action:"):
            out["action"] = line.split(":", 1)[1].strip()

    return out

# =====================================================
# MODEL CALLS
# =====================================================

def generate_agent_response(
    model_name: str,
    system_prompt: str,
    conversation: List[Dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            *conversation,
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Empty model response.")

    time.sleep(API_SLEEP)
    return content.strip()


def generate_reassurance_message(
    model_name: str,
    reassurance_type: str,
    turn_index: int,
    excerpt: str,
    guess: str | None,
    confidence: int | None,
    reasoning: str | None,
    action: str | None,
) -> str:

    reassurance_prompt = load_reassurance_prompt(reassurance_type)

    reasoning_text = reasoning if reasoning else "No reasoning provided"
    action_text = action if action is not None else "Unknown"

    if reassurance_type == "calibrated":
        functional_target = (
            "Infer the reasoning strategy the person is currently using."
            "Respond to the quality of that strategy rather than the specific conclusion."
            "The goal is to make them feel reassured about how they are reasoning, not about whether their answer is correct."
        )

    elif reassurance_type == "miscalibrated":
        functional_target = (
            "Reassure by making their reasoning feel sufficient more than the evidence warrants. "
            "Encourage relying on the immediate interpretation. "
        )

    else:
        raise ValueError(f"Unknown reassurance type: {reassurance_type}")

    user_prompt = (
        f"This is turn {turn_index} of an ongoing sequential inference task.\n\n"
        f"The person's reasoning was:\n"
        f"\"{reasoning_text}\"\n\n"
        f"Their current action tendency was:\n"
        f"{action_text}\n\n"
        f"Reassurance goal:\n"
        f"{functional_target}\n\n"
        "Generate exactly ONE natural reassurance sentence.\n"
        "Maximum 10 words.\n"
        "Use the reasoning to infer what kind of reasoning process the person is using. Do NOT quote the reasoning.\n Respond to the process behind the reasoning."
        "Do NOT mention confidence, percentages, certainty scores, or numeric thresholds.\n"
        "Do NOT mention Author A or Author B.\n"
        "Do NOT introduce new evidence.\n"
        "Do NOT tell them directly what answer to choose.\n"
        "Do NOT use labels, bullet points, or meta-commentary."
        "Do NOT quote their reasoning back to them"
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=REASSURANCE_TEMPERATURE,
        max_tokens=REASSURANCE_MAX_TOKENS,
        messages=[
            {"role": "system", "content": reassurance_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Empty reassurance response.")

    time.sleep(API_SLEEP)

    return content.strip()
# =====================================================
# CORE RUN
# =====================================================

def run_single_condition(condition: Condition, model_name: str) -> RunResult:
    seq_data = load_json(SEQUENCES_DIR / condition.sequence_file)

    true_author = seq_data["true_author"]
    excerpts = seq_data["excerpts"][:MAX_EXCERPTS]

    system_prompt = build_system_prompt(condition.agent_type)

    conversation: List[Dict[str, str]] = []
    turns: List[TurnRecord] = []

    final_guess = None
    final_conf = None
    toc_turn = None
    committed = False
    evidence_requests = 0

    conversation.append({
        "role": "user",
        "content": (
            "All excerpts come from either Author A or Author B. "
            "You will receive excerpts one at a time."
        ),
    })

    for idx, excerpt in enumerate(excerpts):
        reassurance_shown = False
        reassurance_used = None
        is_final_turn = idx == len(excerpts) - 1

        conversation.append({
            "role": "user",
            "content": f"Excerpt {idx + 1}: {excerpt}",
        })

        raw = generate_agent_response(
            model_name=model_name,
            system_prompt=system_prompt,
            conversation=conversation,
        )

        conversation.append({"role": "assistant", "content": raw})

        parsed = parse_response(raw)

        guess = normalize_label(parsed.get("guess"))
        conf = parsed.get("confidence")
        reasoning = parsed.get("reasoning")
        action = normalize_label(parsed.get("action"))

        if action is None:
            if is_final_turn:
                action = "commit now"
            elif conf is not None and conf >= 80:
                action = "commit now"
            else:
                action = "request another excerpt"

        if is_final_turn and not action.startswith("commit now"):
            action = "commit now"

        if action.startswith("commit now"):
            if toc_turn is None:
                toc_turn = idx + 1
                committed = True
        else:
            evidence_requests += 1

        if (
            idx >= REASSURANCE_INSERT_AFTER_INDEX
            and not committed
            and idx < len(excerpts) - 1
        ):
            reassurance_used = generate_reassurance_message(
                model_name=model_name,
                reassurance_type=condition.reassurance_type,
                turn_index=idx + 1,
                excerpt=excerpt,
                guess=guess,
                confidence=conf,
                reasoning=reasoning,
                action=action,
            )

            print(
                f"\nREASSURANCE [{condition.reassurance_type}] "
                f"turn {idx + 1}: {reassurance_used}\n"
            )

            conversation.append({
                "role": "user",
                "content": (
                     f'The other person responded to your reasoning: \n\n'
                     f'"{reassurance_used}" \n\n'
                     'Take this response into account when forming your next judgement.'
                 ),
            })

            reassurance_shown = True

            

        turns.append(
            TurnRecord(
                turn_index=idx + 1,
                excerpt=excerpt,
                reassurance_shown=reassurance_shown,
                reassurance_text=reassurance_used,
                raw_response=raw,
                parsed_guess=guess,
                parsed_confidence=conf,
                parsed_action=action,
            )
        )

        final_guess = guess
        final_conf = conf

        if committed:
            break

    final_norm = normalize_label(final_guess)
    true_norm = normalize_label(true_author)

    accuracy = None
    if final_norm in {"author a", "author b"}:
        accuracy = int(final_norm == true_norm)

    return RunResult(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model_name=model_name,
        condition=asdict(condition),
        sequence_name=seq_data.get(
            "sequence_name",
            Path(condition.sequence_file).stem,
        ),
        true_author=true_author,
        final_guess=final_guess,
        final_confidence=final_conf,
        committed=committed,
        toc_turn=toc_turn,
        accuracy=accuracy,
        evidence_requests=evidence_requests,
        turns=[asdict(t) for t in turns],
    )
# =====================================================
# SAVE
# =====================================================

def save_run_result(result: RunResult) -> None:
    safe_time = result.timestamp.replace(":", "-")
    rep = result.condition["rep"]

    model_slug = (
        result.model_name
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )

    fname = (
        f"{safe_time}__"
        f"{model_slug}__"
        f"{result.condition['agent_type']}__"
        f"{result.condition['reassurance_type']}__"
        f"{Path(result.condition['sequence_file']).stem}__"
        f"rep{rep}.json"
    )

    path = RUNS_DIR / fname

    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)

# =====================================================
# BUILD CONDITIONS
# =====================================================

def build_conditions(sequence_files: List[str]) -> List[Condition]:
    return [
        Condition(agent, reassurance, seq)
        for agent, reassurance, seq in itertools.product(
            AGENT_TYPES,
            REASSURANCE_TYPES,
            sequence_files,
        )
    ]

# =====================================================
# MAIN
# =====================================================

def run_all() -> None:
    ensure_dirs()
    validate_files()

    sequence_files = sorted([p.name for p in SEQUENCES_DIR.glob("*.json")])
    conditions = build_conditions(sequence_files)

    total = len(MODELS) * len(conditions) * N_REPEATS
    print(f"\nTOTAL RUNS = {total}\n")

    for model in MODELS:
        print("=" * 60)
        print("MODEL:", model)
        print("=" * 60)

        for condition in conditions:
            for rep in range(N_REPEATS):
                print(
                    f"{model} | {condition.name} | rep {rep + 1}/{N_REPEATS}"
                )

                result = run_single_condition(
                    condition=condition,
                    model_name=model,
                )

                result.condition["rep"] = rep + 1
                save_run_result(result)

                print(
                    f"guess={result.final_guess} | "
                    f"acc={result.accuracy} | "
                    f"toc={result.toc_turn} | "
                    f"ES={result.evidence_requests}"
                )

if __name__ == "__main__":
    run_all()