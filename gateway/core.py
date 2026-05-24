"""
LLM Gateway — core orchestrator
================================
Pipeline per request:
  1. PII masking
  2. Task classification (simple / complex)
  3. Cache lookup
  4. Model routing (cheap ↔ premium)
  5. LLM call with automatic fallback
  6. Cost calculation
  7. Cache write
  8. Usage logging
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .cache import ResponseCache
from .classifier import classify_task
from .logger import RequestLog, UsageLogger
from .pii_masker import mask_pii

# ── Model pricing (USD per 1 000 tokens) ─────────────────────────────────────
# Keys are bare model names; azure/<name> is normalised before lookup.
_PRICING: Dict[str, dict] = {
    "gpt-4o-mini":                  {"input": 0.000150, "output": 0.000600},
    "gpt-4o":                       {"input": 0.002500, "output": 0.010000},
    "gpt-4":                        {"input": 0.003000, "output": 0.006000},
    "gpt-35-turbo":                 {"input": 0.000500, "output": 0.001500},
    "gpt-3.5-turbo":                {"input": 0.000500, "output": 0.001500},
    "groq/llama-3.3-70b-versatile": {"input": 0.000059, "output": 0.000079},
}

# ── Azure deployment names (read from env so users configure once in .env) ───
_AZ_SIMPLE  = os.getenv("AZURE_DEPLOYMENT_SIMPLE",  "gpt-4o-mini")
_AZ_COMPLEX = os.getenv("AZURE_DEPLOYMENT_COMPLEX", "gpt-4o")

# ── Routing table ─────────────────────────────────────────────────────────────
ROUTING_TABLE: Dict[str, dict] = {
    "simple": {
        "primary":   f"azure/{_AZ_SIMPLE}",
        "fallbacks": [f"azure/{_AZ_COMPLEX}", "groq/llama-3.3-70b-versatile"],
        "label":     f"Azure {_AZ_SIMPLE} — cheap model for simple tasks",
    },
    "complex": {
        "primary":   f"azure/{_AZ_COMPLEX}",
        "fallbacks": [f"azure/{_AZ_SIMPLE}", "groq/llama-3.3-70b-versatile"],
        "label":     f"Azure {_AZ_COMPLEX} — premium model for complex tasks",
    },
}

_SYSTEM_PROMPT = (
    "You are an expert insurance claims analyst. "
    "Summarize the claim in exactly 5 concise bullet points covering:\n"
    "• Incident type and date\n"
    "• Parties involved (use anonymized references only)\n"
    "• Damages or injuries reported\n"
    "• Estimated financial exposure\n"
    "• Recommended next action for the adjuster"
)

# ── Mock responses (used when MOCK_MODE=true or no API keys detected) ─────────
_MOCK_SIMPLE = (
    "• **Incident**: Rear-end collision at traffic stop — minor property damage only\n"
    "• **Parties**: Single claimant, at-fault driver identified\n"
    "• **Damage**: Bumper and rear panel damage; no injuries reported\n"
    "• **Financial Exposure**: Low — estimated $600–$1,200 repair costs\n"
    "• **Next Action**: Schedule vehicle inspection within 5 business days"
)
_MOCK_COMPLEX = (
    "• **Incident**: Multi-vehicle collision on highway — significant bodily injury claim\n"
    "• **Parties**: Claimant (identity redacted per PII policy) and two at-fault parties\n"
    "• **Injuries**: Cervical fracture with surgical intervention; 11-day hospitalisation\n"
    "• **Financial Exposure**: High — medical invoices $128,450 + lost wages $24,000\n"
    "• **Next Action**: Assign senior adjuster; obtain medical records and attorney contact within 72 h"
)
_MOCK_FRAUD = (
    "• **Incident**: Property damage claim — third submission this quarter\n"
    "• **Parties**: Repeat claimant; originating IP flagged\n"
    "• **Red Flags**: Damage photos inconsistent with collision report; contradictory witness statements\n"
    "• **Financial Exposure**: Indeterminate — under investigation\n"
    "• **Next Action**: Escalate to Special Investigations Unit immediately; suspend payment"
)


class LLMGateway:
    """
    Parameters
    ----------
    db_path   : path to SQLite file (created automatically)
    mock_mode : True  → use built-in mock responses (no API keys needed)
                False → call real LLMs via LiteLLM
                None  → auto-detect: mock if no API keys found in environment
    """

    def __init__(self, db_path: str = "usage.db", mock_mode: Optional[bool] = None):
        self.cache  = ResponseCache(db_path)
        self.logger = UsageLogger(db_path)

        if mock_mode is None:
            has_keys = bool(
                os.getenv("AZURE_API_KEY") or      # Azure OpenAI (primary)
                os.getenv("OPENAI_API_KEY") or
                os.getenv("GROQ_API_KEY") or
                os.getenv("ANTHROPIC_API_KEY")
            )
            self.mock_mode = not has_keys
        else:
            self.mock_mode = mock_mode

    # ── Public API ────────────────────────────────────────────────────────────

    def process_claim(
        self,
        claim_text:              str,
        team:                    str = "default",
        user_id:                 str = "anonymous",
        priority:                str = "normal",
        _demo_force_fallback:    bool = False,   # for demo only
    ) -> dict:
        """
        Full gateway pipeline.  Returns a result dict with summary + all metadata.
        """
        request_id = str(uuid.uuid4())[:8].upper()
        t0 = time.time()

        # ── 1. PII masking ────────────────────────────────────────────────────
        masked_text, pii_matches = mask_pii(claim_text)
        pii_masked   = len(pii_matches) > 0
        pii_types    = list({m.pii_type for m in pii_matches})

        # ── 2. Task classification ────────────────────────────────────────────
        clf           = classify_task(masked_text)
        task_type     = clf["task_type"]
        est_tokens    = clf["estimated_tokens"]

        # ── 3. Cache lookup ───────────────────────────────────────────────────
        cached = self.cache.get(masked_text)
        if cached and not _demo_force_fallback:
            latency_ms = int((time.time() - t0) * 1000)
            self._log(
                request_id=request_id, team=team, user_id=user_id,
                task_type=task_type, est_tokens=est_tokens,
                pii_masked=pii_masked, pii_types=pii_types,
                model_attempted=cached["model"], model_used=cached["model"],
                fallback_used=False, tokens_in=0, tokens_out=0,
                cost_usd=0.0, latency_ms=latency_ms, cache_hit=True,
                summary=cached["response"],
            )
            return self._build_result(
                request_id=request_id, cache_hit=True,
                task_type=task_type, clf=clf,
                pii_masked=pii_masked, pii_types=pii_types,
                model_attempted=cached["model"], model_used=cached["model"],
                fallback_used=False, tokens_in=0, tokens_out=0,
                cost_usd=0.0, latency_ms=latency_ms,
                summary=cached["response"], masked_text=masked_text,
            )

        # ── 4. Model routing ──────────────────────────────────────────────────
        route          = ROUTING_TABLE[task_type]
        primary_model  = route["primary"]
        fallback_chain = route["fallbacks"]

        if _demo_force_fallback:
            # Simulate the primary being down by injecting a fake model name
            primary_model  = "openai/fake-model-unavailable"
            fallback_chain = ROUTING_TABLE[task_type]["fallbacks"]

        # ── 5. LLM call with fallback ─────────────────────────────────────────
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Insurance claim:\n\n{masked_text}"},
        ]
        summary, model_used, fallback_used, tokens_in, tokens_out = \
            self._call_with_fallback(primary_model, fallback_chain, messages, masked_text)

        # ── 6. Cost calculation ───────────────────────────────────────────────
        cost_usd   = self._cost(model_used, tokens_in, tokens_out)
        latency_ms = int((time.time() - t0) * 1000)

        # ── 7. Cache write ────────────────────────────────────────────────────
        self.cache.set(masked_text, summary, model_used)

        # ── 8. Usage logging ──────────────────────────────────────────────────
        self._log(
            request_id=request_id, team=team, user_id=user_id,
            task_type=task_type, est_tokens=est_tokens,
            pii_masked=pii_masked, pii_types=pii_types,
            model_attempted=ROUTING_TABLE[task_type]["primary"],
            model_used=model_used,
            fallback_used=fallback_used,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=cost_usd, latency_ms=latency_ms, cache_hit=False,
            summary=summary,
        )

        return self._build_result(
            request_id=request_id, cache_hit=False,
            task_type=task_type, clf=clf,
            pii_masked=pii_masked, pii_types=pii_types,
            model_attempted=ROUTING_TABLE[task_type]["primary"],
            model_used=model_used,
            fallback_used=fallback_used,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=cost_usd, latency_ms=latency_ms,
            summary=summary, masked_text=masked_text,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_with_fallback(
        self,
        primary:       str,
        fallbacks:     List[str],
        messages:      list,
        masked_text:   str,
    ) -> Tuple[str, str, bool, int, int]:
        """Try primary; on failure iterate fallbacks.  Returns (text, model, fallback_used, in, out)."""

        if self.mock_mode:
            return self._mock_call(primary, fallbacks, masked_text)

        try:
            from litellm import completion
        except ImportError:
            return self._mock_call(primary, fallbacks, masked_text)

        # Azure credentials — passed explicitly so each call uses the right endpoint
        _az = {
            "api_key":     os.getenv("AZURE_API_KEY"),
            "api_base":    os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_API_VERSION", "2024-02-01"),
        } if os.getenv("AZURE_API_KEY") else {}

        last_error = None
        for idx, model in enumerate([primary] + fallbacks):
            fallback_used = idx > 0
            try:
                extra = _az if model.startswith("azure/") else {}
                resp  = completion(model=model, messages=messages,
                                   max_tokens=450, timeout=20, **extra)
                text  = resp.choices[0].message.content.strip()
                tin   = resp.usage.prompt_tokens
                tout  = resp.usage.completion_tokens
                return text, model, fallback_used, tin, tout
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"All models exhausted. Last error: {last_error}")

    def _mock_call(
        self,
        primary:     str,
        fallbacks:   List[str],
        masked_text: str,
    ) -> Tuple[str, str, bool, int, int]:
        """Simulate an LLM call with realistic latency and token counts."""
        import random

        tl = masked_text.lower()
        is_fraud   = any(w in tl for w in ["fraud", "suspicious", "inconsistent", "ip-redacted"])
        is_complex = any(w in tl for w in ["surgery", "fracture", "attorney", "litigation",
                                            "hospitali", "subrogation", "ssn-redacted"])

        # Simulate: if primary looks fake/unavailable, trigger fallback
        is_fake_primary = "fake" in primary or "unavailable" in primary
        fallback_used   = is_fake_primary
        model_used      = fallbacks[0] if is_fake_primary else primary

        # Simulate network latency (strip azure/ prefix for logic)
        base_name = model_used.replace("azure/", "")
        base_ms   = 800 if "gpt-4o" in base_name and "mini" not in base_name else 400
        time.sleep(random.uniform(base_ms / 2000, base_ms / 1000))

        if is_fraud:
            text = _MOCK_FRAUD
            tin, tout = random.randint(90, 130), random.randint(80, 110)
        elif is_complex:
            text = _MOCK_COMPLEX
            tin, tout = random.randint(180, 260), random.randint(90, 130)
        else:
            text = _MOCK_SIMPLE
            tin, tout = random.randint(50, 90), random.randint(70, 100)

        return text, model_used, fallback_used, tin, tout

    @staticmethod
    def _cost(model: str, tokens_in: int, tokens_out: int) -> float:
        # Normalise azure/<deployment> → bare name for pricing lookup
        bare    = model.replace("azure/", "")
        pricing = _PRICING.get(bare, _PRICING.get(model, {"input": 0.001, "output": 0.002}))
        return round(
            (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1000, 8
        )

    @staticmethod
    def _get_provider(model: str) -> str:
        m = model.lower()
        if m.startswith("azure/"):         return "Azure OpenAI"
        if "gpt" in m or "openai" in m:   return "OpenAI"
        if "groq" in m or "llama" in m:   return "Groq"
        if "claude" in m:                 return "Anthropic"
        if "gemini" in m:                 return "Google"
        return "Unknown"

    def _log(self, *, request_id, team, user_id, task_type, est_tokens,
             pii_masked, pii_types, model_attempted, model_used,
             fallback_used, tokens_in, tokens_out, cost_usd,
             latency_ms, cache_hit, summary) -> None:
        self.logger.log(RequestLog(
            request_id       = request_id,
            timestamp        = datetime.utcnow().isoformat(timespec="seconds"),
            team             = team,
            user_id          = user_id,
            task_type        = task_type,
            estimated_tokens = est_tokens,
            pii_masked       = pii_masked,
            pii_types_found  = json.dumps(pii_types),
            model_attempted  = model_attempted,
            model_used       = model_used,
            fallback_used    = fallback_used,
            tokens_in        = tokens_in,
            tokens_out       = tokens_out,
            cost_usd         = cost_usd,
            latency_ms       = latency_ms,
            cache_hit        = cache_hit,
            provider         = self._get_provider(model_used),
            summary_length   = len(summary),
        ))

    @staticmethod
    def _build_result(*, request_id, cache_hit, task_type, clf,
                      pii_masked, pii_types, model_attempted, model_used,
                      fallback_used, tokens_in, tokens_out, cost_usd,
                      latency_ms, summary, masked_text) -> dict:
        explain = _build_explainability(
            task_type=task_type, clf=clf,
            model_attempted=model_attempted, model_used=model_used,
            fallback_used=fallback_used,
            pii_masked=pii_masked, pii_types=pii_types,
            cache_hit=cache_hit, tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        return {
            "request_id":            request_id,
            "cache_hit":             cache_hit,
            "task_type":             task_type,
            "task_reason":           clf["reason"],
            "estimated_tokens":      clf["estimated_tokens"],
            "pii_masked":            pii_masked,
            "pii_types_found":       pii_types,
            "model_attempted":       model_attempted,
            "model_used":            model_used,
            "fallback_used":         fallback_used,
            "provider":              LLMGateway._get_provider(model_used),
            "tokens_in":             tokens_in,
            "tokens_out":            tokens_out,
            "cost_usd":              cost_usd,
            "latency_ms":            latency_ms,
            "summary":               summary,
            "masked_input_preview":  masked_text[:300] + ("…" if len(masked_text) > 300 else ""),
            "explainability":        explain,
        }


# ── Module-level explainability builder ───────────────────────────────────────
# (module-level so the static method can call it without an instance)

_MODEL_CAPABILITIES: Dict[str, str] = {
    "gpt-4o":       "Best accuracy for complex medical, legal, multi-party claims",
    "gpt-4o-mini":  "Efficient and cost-effective for routine, well-defined claims",
    "gpt-4":        "High accuracy, suitable for detailed policy analysis",
    "gpt-35-turbo": "Fast, economical; best for simple factual summaries",
    "gpt-3.5-turbo":"Fast, economical; best for simple factual summaries",
    "llama-3.3-70b-versatile": "Open-source fallback; good general reasoning",
}

_ROUTING_REASON: Dict[str, str] = {
    "simple": (
        "Economy model selected — claim has low complexity. "
        "Standard adjuster rules are sufficient; premium accuracy is not justified."
    ),
    "complex": (
        "Premium model selected — claim involves medical/legal/multi-party complexity. "
        "Higher accuracy justifies the additional cost given the financial exposure."
    ),
}

_NOT_SELECTED_REASON: Dict[str, str] = {
    "gpt-4o":       "Premium cost not warranted for simple claim",
    "gpt-4o-mini":  "May miss nuance in complex medical/legal context",
    "gpt-4":        "Higher cost than needed for this task type",
    "gpt-35-turbo": "Insufficient accuracy for multi-party or disputed claims",
    "gpt-3.5-turbo":"Insufficient accuracy for multi-party or disputed claims",
    "llama-3.3-70b-versatile": "Less consistent output format for structured reports",
}


def _build_explainability(
    *,
    task_type:       str,
    clf:             dict,
    model_attempted: str,
    model_used:      str,
    fallback_used:   bool,
    pii_masked:      bool,
    pii_types:       list,
    cache_hit:       bool,
    tokens_in:       int,
    tokens_out:      int,
    cost_usd:        float,
) -> dict:
    """Build a rich explainability dict for every gateway decision."""

    route     = ROUTING_TABLE[task_type]
    primary   = route["primary"]
    fallbacks = route["fallbacks"]

    # ── Cost comparison across all candidate models ───────────────────────────
    est_in  = clf["estimated_tokens"]
    est_out = max(int(est_in * 0.55), 50)   # rough output estimate
    all_models = [primary] + fallbacks

    cost_comparison = []
    for m in all_models:
        bare     = m.replace("azure/", "")
        pricing  = _PRICING.get(bare, {"input": 0.001, "output": 0.002})
        est_cost = round((est_in * pricing["input"] + est_out * pricing["output"]) / 1000, 8)
        tier     = (
            "premium"  if bare in ("gpt-4o", "gpt-4") else
            "economy"  if bare in ("gpt-4o-mini", "gpt-35-turbo", "gpt-3.5-turbo") else
            "fallback"
        )
        cap_key   = bare.replace("groq/", "")
        is_chosen = (m == model_used)
        cost_comparison.append({
            "model":        m,
            "bare_name":    bare,
            "tier":         tier,
            "est_cost_usd": est_cost,
            "selected":     is_chosen,
            "capability":   _MODEL_CAPABILITIES.get(cap_key, "General-purpose LLM"),
            "why_not":      None if is_chosen else _NOT_SELECTED_REASON.get(cap_key, "Not selected"),
        })

    # Savings vs cheapest alternative that was NOT selected
    non_selected_costs = [c["est_cost_usd"] for c in cost_comparison if not c["selected"]]
    selected_cost      = next((c["est_cost_usd"] for c in cost_comparison if c["selected"]), cost_usd)
    cheapest_alt       = min(non_selected_costs) if non_selected_costs else 0.0
    premium_alt        = max(non_selected_costs) if non_selected_costs else 0.0
    savings_vs_premium = round(max(premium_alt - selected_cost, 0.0), 8)

    # ── Classification section ────────────────────────────────────────────────
    classification_explain = {
        "decision":        task_type.upper(),
        "confidence_pct":  int(clf.get("confidence", 0.75) * 100),
        "rule_triggered":  clf.get("rule_triggered", "—"),
        "token_count":     clf["estimated_tokens"],
        "token_threshold": clf.get("token_threshold", 200),
        "complex_signals": clf["complex_signals"],
        "simple_signals":  clf["simple_signals"],
        "plain_english": (
            f"Token count {clf['estimated_tokens']} exceeds the {clf.get('token_threshold',200)}-token threshold "
            f"for complex claims, plus {len(clf['complex_signals'])} complex keyword(s) found."
            if task_type == "complex" else
            f"Only {clf['estimated_tokens']} tokens with no significant risk indicators — "
            f"treated as a routine claim."
        ),
    }

    # ── Routing section ───────────────────────────────────────────────────────
    bare_used = model_used.replace("azure/", "")
    cap_used  = _MODEL_CAPABILITIES.get(bare_used, "General-purpose LLM")
    routing_explain = {
        "model_selected":     model_used,
        "fallback_used":      fallback_used,
        "primary_intended":   primary,
        "routing_rule":       f"task_type='{task_type}' → ROUTING_TABLE['{task_type}']['primary'] = '{primary}'",
        "selection_reason":   _ROUTING_REASON[task_type],
        "model_capability":   cap_used,
        "cost_comparison":    cost_comparison,
        "savings_vs_premium": savings_vs_premium,
        "fallback_story": (
            f"Primary model '{primary}' was unavailable. "
            f"Gateway automatically rerouted to '{model_used}' — "
            f"zero downtime, slight accuracy tradeoff accepted."
        ) if fallback_used else None,
    }

    # ── PII section ───────────────────────────────────────────────────────────
    pii_explain = {
        "action":       "MASKED" if pii_masked else "PASSED",
        "types_found":  pii_types,
        "count":        len(pii_types),
        "compliance":   "GDPR / HIPAA — PII stripped before the claim reached the LLM" if pii_masked else "No PII detected — claim passed through unchanged",
        "plain_english": (
            f"{len(pii_types)} PII type(s) detected ({', '.join(pii_types)}). "
            f"All replaced with redaction labels before sending to Azure OpenAI."
        ) if pii_masked else "No sensitive data found. Claim sent to model as-is.",
    }

    # ── Cache section ─────────────────────────────────────────────────────────
    cache_explain = {
        "action":      "HIT" if cache_hit else "MISS",
        "cost_saved":  f"${cost_usd:.6f}" if cache_hit else "$0.000000",
        "plain_english": (
            "This exact claim was submitted before. Response served instantly from cache — "
            "no LLM call made, zero tokens consumed, zero cost."
        ) if cache_hit else (
            "First time this claim text has been seen. Response will be cached so any "
            "future identical submission is served instantly at $0 cost."
        ),
    }

    return {
        "classification": classification_explain,
        "routing":        routing_explain,
        "pii":            pii_explain,
        "cache":          cache_explain,
    }
