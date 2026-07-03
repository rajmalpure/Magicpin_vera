#!/usr/bin/env python3
"""
generate_submission.py
======================
Runs all seed triggers through the bot's local composer and writes
submission.jsonl — 30 lines, one per test pair.

Usage:
    python generate_submission.py

Output: submission.jsonl (in the project root)
"""

import json
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Load dataset ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATASET = BASE / "dataset"


def load_dataset():
    categories = {}
    merchants = {}
    customers = {}
    triggers = {}

    # Categories
    cat_dir = DATASET / "categories"
    for f in cat_dir.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        categories[data["slug"]] = data

    # Merchants
    m_path = DATASET / "merchants_seed.json"
    m_data = json.loads(m_path.read_text(encoding="utf-8"))
    for m in m_data["merchants"]:
        merchants[m["merchant_id"]] = m

    # Customers
    c_path = DATASET / "customers_seed.json"
    c_data = json.loads(c_path.read_text(encoding="utf-8"))
    for c in c_data["customers"]:
        merchants_key = c.get("customer_id", c.get("id"))
        if merchants_key:
            customers[merchants_key] = c

    # Triggers
    t_path = DATASET / "triggers_seed.json"
    t_data = json.loads(t_path.read_text(encoding="utf-8"))
    for t in t_data["triggers"]:
        triggers[t["id"]] = t

    return categories, merchants, customers, triggers


def main():
    # Check API key
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if not any([groq_key, openai_key, gemini_key]):
        logger.error("No LLM API key found! Set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")
        sys.exit(1)

    from composer import EngagementComposer
    comp = EngagementComposer()
    logger.info(f"Using LLM provider: {comp.llm_provider}")

    categories, merchants, customers, triggers = load_dataset()
    logger.info(f"Loaded: {len(categories)} categories, {len(merchants)} merchants, {len(customers)} customers, {len(triggers)} triggers")

    output_path = BASE / "submission.jsonl"
    lines = []
    test_id = 1

    for trg_id, trg in triggers.items():
        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        merchant = merchants.get(merchant_id)
        if not merchant:
            logger.warning(f"T{test_id:02d}: merchant {merchant_id} not found — skipping {trg_id}")
            continue

        cat_slug = merchant.get("category_slug")
        category = categories.get(cat_slug)
        if not category:
            logger.warning(f"T{test_id:02d}: category {cat_slug} not found — skipping {trg_id}")
            continue

        customer = customers.get(customer_id) if customer_id else None

        logger.info(f"T{test_id:02d}: Composing for {trg_id} → {merchant.get('identity', {}).get('name')}")
        try:
            msg = comp.compose(category, merchant, trg, customer)
            lines.append({
                "test_id": f"T{test_id:02d}",
                "trigger_id": trg_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "body": msg.body,
                "cta": msg.cta,
                "send_as": msg.send_as,
                "suppression_key": msg.suppression_key,
                "rationale": msg.rationale,
            })
            test_id += 1
            # Polite rate-limiting for Groq free tier
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"T{test_id:02d}: Error — {e}")
            test_id += 1

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info(f"\n✅ Written {len(lines)} entries to {output_path}")

    # Print first entry as a sample
    if lines:
        print("\n=== SAMPLE (first entry) ===")
        first = lines[0]
        print(f"test_id: {first['test_id']}")
        print(f"trigger_id: {first['trigger_id']}")
        print(f"send_as: {first['send_as']}")
        print(f"cta: {first['cta']}")
        print(f"body:\n{first['body']}")
        print(f"rationale: {first['rationale']}")


if __name__ == "__main__":
    main()
