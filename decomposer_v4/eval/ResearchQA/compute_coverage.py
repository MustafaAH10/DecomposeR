import argparse
import json
from tqdm import tqdm
from typing import List
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from openai import OpenAI

try:
    from .researchqa_loader import load_researchqa_data
except ImportError:
    from researchqa_loader import load_researchqa_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client: OpenAI | None = None

# Custom exception for batch processing failures
class BatchProcessingError(Exception):
    pass

def call_gpt(prompt: str, model: str = "gpt-4") -> str:
    global client
    if client is None:
        client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()

def build_prompt(response: str, questions: List[str]) -> str:
    prompt = (
        "Please judge the following questions based on the response below.\n"
        "For each question, select one of the following ratings to indicate the extent to which the response addresses the question:\n"
        "Not at all, Barely, Moderately, Mostly, Completely\n\n"
        "Definitions:\n"
        "- Not at all: *totally uninferable*\n"
        "- Barely: *unmentioned but inferrable*\n"
        "- Moderately: *mentioned but misses important details*\n"
        "- Mostly: *mentioned but misses some details*\n"
        "- Completely: *mentioned with sufficient details*\n\n"
        "Only output one of the five phrases for each question, separated by newlines, and nothing else.\n\n"
        f"Response: {response}\n"
        f"Questions:\n" + '\n'.join(questions) + "\n\nOutput:"
    )
    return prompt

CANONICAL_LABELS = (
    "Not at all",
    "Barely",
    "Moderately",
    "Mostly",
    "Completely",
)


def normalize_score_label(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[\).\:-])\s*", "", cleaned).strip()
    cleaned = cleaned.strip("\"'`.,;:")
    lowered = cleaned.lower()
    for label in CANONICAL_LABELS:
        if lowered == label.lower():
            return label
    for label in CANONICAL_LABELS:
        if re.search(rf"\b{re.escape(label.lower())}\b", lowered):
            return label
    raise ValueError(f"Unrecognized coverage label: {text!r}")


def convert_text_to_score(text):
    text_to_score = {
        'Not at all': 1,
        'Barely': 2,
        'Moderately': 3,
        'Mostly': 4,
        'Completely': 5
    }
    score = text_to_score[normalize_score_label(text)]
    return score

def normalize_5scale(x):
    return (x - 1) / 4

def compute_coverage_score(rubric_judges):
    """
    rubric_judges: list of dicts, each with a 'score' key (text label)
    Returns: average normalized coverage score (float) in [0, 1]
    """
    numeric_scores = [convert_text_to_score(j['score']) for j in rubric_judges]
    normalized_scores = [normalize_5scale(x) for x in numeric_scores]
    return sum(normalized_scores) / len(normalized_scores)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _empty_group() -> dict:
    return {
        "item_coverages": [],
        "rubric_coverages": [],
        "items": 0,
        "rubric_items": 0,
    }


def _finalize_group(row: dict) -> dict:
    item_coverages = list(row.get("item_coverages", []))
    rubric_coverages = list(row.get("rubric_coverages", []))
    return {
        "coverage": _mean(item_coverages),
        "item_average_coverage": _mean(item_coverages),
        "rubric_average_coverage": _mean(rubric_coverages),
        "items": int(row.get("items", 0)),
        "rubric_items": int(row.get("rubric_items", 0)),
    }

def process_batch_with_retry(prompt: str, batch: List[str], model: str, item_id: str) -> List[str]:
    """
    Process a batch of rubric items with retry logic.

    Args:
        prompt: The prompt to send to GPT
        batch: List of rubric items in the batch
        model: The model name for logging purposes
        item_id: The item ID for logging purposes

    Returns:
        List of scores for the batch
    """
    delay_seconds = 1.0
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            gpt_output = call_gpt(prompt, model=model)
            scores = gpt_output.split('\n')

            # Filter out empty lines
            scores = [score.strip() for score in scores if score.strip()]

            if len(scores) == len(batch):
                try:
                    return [normalize_score_label(score) for score in scores]
                except ValueError as exc:
                    logger.warning(f"Invalid coverage label for item {item_id}: {exc}. Retrying...")
                    last_error = exc
                    continue

            logger.warning(f"Expected {len(batch)} results for item {item_id}, got {len(scores)}. Retrying...")
            logger.warning(f"Questions: {batch}")
            last_error = BatchProcessingError(f"Expected {len(batch)} results, got {len(scores)}")
        except Exception as exc:
            last_error = exc
            logger.warning(f"Batch retry {attempt}/3 failed for item {item_id}: {exc}")
        if attempt < 3:
            time.sleep(delay_seconds)
            delay_seconds *= 2.0
    if isinstance(last_error, Exception):
        raise last_error
    raise BatchProcessingError(f"Unknown batch failure for item {item_id}")

def compute_coverage(
    data_path: str,
    response_map_path: str,
    output_path: str,
    batch_size: int = 8,
    model: str = "gpt-4.1-mini"
):
    # Load questions
    items = load_researchqa_data(data_path)
    # Load responses
    with open(response_map_path, 'r', encoding='utf-8') as f:
        response_map = json.load(f)
    item_results = {}
    item_coverages = []
    rubric_coverages = []
    skipped_items = []
    by_general_domain = defaultdict(_empty_group)
    by_subdomain = defaultdict(_empty_group)
    by_field = defaultdict(_empty_group)
    for item in tqdm(items, desc="Computing rubric coverage"):
        item_id = item.id
        if item_id not in response_map:
            logger.warning(f"Skipping item {item_id} as it has no corresponding response")
            skipped_items.append({"id": item_id, "reason": "missing_response"})
            continue
        raw_response = response_map[item_id]
        if isinstance(raw_response, str):
            answer = raw_response
        elif isinstance(raw_response, dict):
            answer = str(raw_response.get("answer", ""))
        else:
            answer = ""
        rubric_items = [r.rubric_item for r in item.rubric]
        if not rubric_items:
            logger.warning(f"Skipping item {item_id} because it has no rubric items")
            skipped_items.append({"id": item_id, "reason": "missing_rubric"})
            continue
        rubric_judges = []
        item_processing_failed = False

        for i in range(0, len(rubric_items), batch_size):
            batch = rubric_items[i:i+batch_size]
            prompt = build_prompt(answer, batch)

            try:
                # Process batch with retry
                scores = process_batch_with_retry(prompt, batch, model, item_id)

                # Add results to rubric_judges
                for rubric, score in zip(batch, scores):
                    rubric_judges.append({
                        'rubric': rubric,
                        'score': score,
                    })

            except Exception as e:
                logger.error(f"Failed to get correct number of results for batch after all retries. Skipping item {item_id}. Error: {e}")
                item_processing_failed = True
                skipped_items.append({"id": item_id, "reason": f"judge_failure:{e}"})
                break

        # Skip this item if any batch failed
        if item_processing_failed:
            continue

        item_results[item_id] = {
            "id": item_id,
            "general_domain": item.general_domain,
            "subdomain": item.subdomain,
            "field": item.field,
            "rubric_judgments": rubric_judges,
        }
        # Compute and print coverage for this item
        item_coverage = compute_coverage_score(rubric_judges)
        item_results[item_id]["coverage"] = item_coverage
        normalized_rubric_scores = [normalize_5scale(convert_text_to_score(row["score"])) for row in rubric_judges]
        item_coverages.append(item_coverage)
        rubric_coverages.extend(normalized_rubric_scores)
        for bucket_name, bucket in (
            (item.general_domain or "Unknown", by_general_domain),
            (item.subdomain or "Unknown", by_subdomain),
            (item.field or "Unknown", by_field),
        ):
            bucket_row = bucket[bucket_name]
            bucket_row["item_coverages"].append(item_coverage)
            bucket_row["rubric_coverages"].extend(normalized_rubric_scores)
            bucket_row["items"] += 1
            bucket_row["rubric_items"] += len(normalized_rubric_scores)
        logger.info(f"Coverage for {item_id}: {item_coverage:.3f}")

    coverage_payload = {
        "metadata": {
            "data_path": data_path,
            "response_map_path": response_map_path,
            "model": model,
            "batch_size": batch_size,
            "dataset_items": len(items),
            "scored_items": len(item_results),
            "skipped_items": len(skipped_items),
        },
        "overall": {
            "coverage": _mean(item_coverages),
            "item_average_coverage": _mean(item_coverages),
            "rubric_average_coverage": _mean(rubric_coverages),
            "items": len(item_results),
            "rubric_items": len(rubric_coverages),
        },
        "by_general_domain": {
            key: _finalize_group(value)
            for key, value in sorted(by_general_domain.items())
        },
        "by_subdomain": {
            key: _finalize_group(value)
            for key, value in sorted(by_subdomain.items())
        },
        "by_field": {
            key: _finalize_group(value)
            for key, value in sorted(by_field.items())
        },
        "skipped_items": skipped_items,
        "items": item_results,
    }
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coverage_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved rubric coverage results to {output_path}")
    # Print overall average coverage
    if item_coverages:
        avg_coverage = sum(item_coverages) / len(item_coverages)
        logger.info(f"Average rubric coverage: {avg_coverage:.3f}")
    else:
        logger.info("No coverage scores computed.")
    return coverage_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute rubric coverage using GPT")
    parser.add_argument('--data', type=str, required=True, help='Path to ResearchQA JSON file')
    parser.add_argument('--responses', type=str, required=True, help='Path to response map JSON file (id -> response dict)')
    parser.add_argument('--output', type=str, required=True, help='Path to save rubric coverage results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for rubric items')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini', help='OpenAI model name')
    args = parser.parse_args()
    compute_coverage(args.data, args.responses, args.output, args.batch_size, args.model)
