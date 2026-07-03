"""
bot.py — Vera HTTP server for the magicpin AI Challenge
5 required endpoints:
  POST /v1/context  — receive context pushes
  POST /v1/tick     — periodic wake-up; bot decides what to send
  POST /v1/reply    — receive merchant/customer reply; respond synchronously
  GET  /v1/healthz  — liveness probe
  GET  /v1/metadata — bot identity
  POST /v1/teardown — (optional) wipe state at end of test
"""

import os
import sys
import json
import time
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from composer import EngagementComposer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

START = time.time()

# ── Shared composer instance ───────────────────────────────────────────────────
composer = EngagementComposer()
logger.info(f"LLM provider: {composer.llm_provider}")

# ── In-memory state ────────────────────────────────────────────────────────────
# (scope, context_id) → {"version": int, "payload": dict}
contexts: Dict[tuple, Dict[str, Any]] = {}

# conversation_id → list of {"from": role, "msg": text}
conversations: Dict[str, List[Dict[str, str]]] = {}

# suppression: track sent suppression keys to avoid duplicates within a session
sent_suppressions: set = set()

DATASET_DIR = Path(__file__).parent.parent / "dataset"


def _load_seed_dataset():
    """Pre-load the seed dataset into memory so /v1/tick works immediately on startup."""
    count = 0

    # Categories
    cat_dir = DATASET_DIR / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                slug = data.get("slug", f.stem)
                contexts[("category", slug)] = {"version": 1, "payload": data}
                count += 1
            except Exception as e:
                logger.warning(f"Could not load category {f.name}: {e}")

    # Merchants
    m_path = DATASET_DIR / "merchants_seed.json"
    if m_path.exists():
        data = json.loads(m_path.read_text(encoding="utf-8"))
        for m in data.get("merchants", []):
            mid = m.get("merchant_id")
            if mid:
                contexts[("merchant", mid)] = {"version": 1, "payload": m}
                count += 1

    # Customers
    c_path = DATASET_DIR / "customers_seed.json"
    if c_path.exists():
        data = json.loads(c_path.read_text(encoding="utf-8"))
        for c in data.get("customers", []):
            cid = c.get("customer_id")
            if cid:
                contexts[("customer", cid)] = {"version": 1, "payload": c}
                count += 1

    # Triggers
    t_path = DATASET_DIR / "triggers_seed.json"
    if t_path.exists():
        data = json.loads(t_path.read_text(encoding="utf-8"))
        for t in data.get("triggers", []):
            tid = t.get("id")
            if tid:
                contexts[("trigger", tid)] = {"version": 1, "payload": t}
                count += 1

    logger.info(f"Seed dataset loaded: {count} contexts pre-warmed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    _load_seed_dataset()
    yield
    # ── Shutdown (nothing to clean up) ──


app = FastAPI(title="Vera Bot", version="2.0.0", lifespan=lifespan)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/healthz")
async def healthz():
    """Liveness probe — returns loaded context counts."""
    counts: Dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    """Bot identity metadata."""
    return {
        "team_name": "Vera- by Raj",
        "team_members": ["Raj Malpure"],
        "model": f"{composer.llm_provider}/llama-3.3-70b-versatile",
        "approach": (
            "Trigger-kind routed prompt composer with verifiable-fact injection, "
            "category voice enforcement, and explicit compulsion-lever application."
        ),
        "contact_email": "bot@magicpin-challenge.com",
        "version": "2.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    """Receive and store a context push. Idempotent by (context_id, version)."""
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {valid_scopes}"}

    key = (body.scope, body.context_id)
    cur = contexts.get(key)

    if cur:
        if cur["version"] > body.version:
            return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
        if cur["version"] == body.version:
            # Idempotent no-op
            return {
                "accepted": True,
                "ack_id": f"ack_{body.context_id}_v{body.version}",
                "stored_at": datetime.now(timezone.utc).isoformat(),
            }

    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody, force: bool = False):
    """
    Periodic wake-up. Bot inspects available triggers and decides what to send.
    Returns up to 20 actions. Must respond within 30s.
    """
    async def process_trigger(trg_id: str) -> Optional[Dict]:
        # Load trigger context
        trg_data = contexts.get(("trigger", trg_id))
        if not trg_data:
            logger.warning(f"Trigger {trg_id} not in contexts — skipping")
            return None
        trg = trg_data["payload"]

        # Resolve suppression
        sup_key = trg.get("suppression_key", "")
        if not force and sup_key and sup_key in sent_suppressions:
            logger.info(f"Suppressed (already sent): {sup_key}")
            return None

        # Resolve merchant
        merchant_id = trg.get("merchant_id")
        if not merchant_id:
            merchant_id = trg.get("payload", {}).get("merchant_id")
        if not merchant_id:
            logger.warning(f"No merchant_id in trigger {trg_id}")
            return None

        merchant_data = contexts.get(("merchant", merchant_id))
        if not merchant_data:
            logger.warning(f"Merchant {merchant_id} not loaded — skipping {trg_id}")
            return None
        merchant = merchant_data["payload"]

        # Resolve category
        category_slug = merchant.get("category_slug")
        if not category_slug:
            category_slug = trg.get("payload", {}).get("category")
        category_data = contexts.get(("category", category_slug))
        if not category_data:
            logger.warning(f"Category {category_slug} not loaded — skipping {trg_id}")
            return None
        category = category_data["payload"]

        # Resolve optional customer
        customer = None
        customer_id = trg.get("customer_id")
        if customer_id:
            cust_data = contexts.get(("customer", customer_id))
            if cust_data:
                customer = cust_data["payload"]

        # Compose — run in thread to avoid blocking event loop
        composed = await asyncio.to_thread(composer.compose, category, merchant, trg, customer)

        # Mark suppression
        if not force and composed.suppression_key:
            sent_suppressions.add(composed.suppression_key)

        # Build unique conversation_id
        ts = int(time.time() * 1000)
        conv_id = f"conv_{merchant_id}_{trg_id}_{ts}"

        return {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v2",
            "template_params": [
                merchant.get("identity", {}).get("owner_first_name", ""),
                composed.body[:40],
                trg_id,
            ],
            "body": composed.body,
            "cta": composed.cta,
            "suppression_key": composed.suppression_key,
            "rationale": composed.rationale,
        }

    # Process triggers sequentially (with a concurrency of 1 and a small delay)
    # to avoid hitting Groq's RPM (Requests Per Minute) rate limits.
    sem = asyncio.Semaphore(1)

    async def process_trigger_sem(trg_id: str) -> Optional[Dict]:
        async with sem:
            res = await process_trigger(trg_id)
            if res is not None:
                # Add a delay between requests to let the Groq rate limits cool down (30 RPM = 2.0s minimum, using 2.5s for safety)
                await asyncio.sleep(2.5)
            return res

    trigger_batch = body.available_triggers[:20]
    tasks = [process_trigger_sem(tid) for tid in trigger_batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    actions = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Trigger processing error: {r}")
        elif r is not None:
            actions.append(r)

    logger.info(f"Tick at {body.now}: {len(trigger_batch)} triggers → {len(actions)} actions")
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Receive a merchant/customer reply. Respond synchronously within 30s.
    """
    conv_hist = conversations.setdefault(body.conversation_id, [])
    conv_hist.append({"from": body.from_role, "msg": body.message})

    # Load merchant context for personalisation
    merchant = {}
    if body.merchant_id:
        md = contexts.get(("merchant", body.merchant_id))
        if md:
            merchant = md["payload"]

    # Delegate to composer
    decision = await asyncio.to_thread(composer.handle_reply, conv_hist, body.message, merchant)

    # Append bot's reply to history
    if decision.get("action") == "send" and decision.get("body"):
        conv_hist.append({"from": "vera", "msg": decision["body"]})

    logger.info(f"Reply in {body.conversation_id} turn={body.turn_number}: action={decision.get('action')}")
    return decision


@app.post("/v1/teardown")
async def teardown():
    """Wipe all in-memory state at end of test (optional endpoint)."""
    contexts.clear()
    conversations.clear()
    sent_suppressions.clear()
    logger.info("State wiped via teardown")
    return {"status": "cleared"}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False, log_level="info")
