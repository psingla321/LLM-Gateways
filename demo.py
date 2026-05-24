#!/usr/bin/env python3
"""
LLM Gateway Demo — Insurance Claim Processing
==============================================
Five live scenarios demonstrating every gateway feature:

  Demo 1 ─ Simple claim        → cheap model (gpt-4o-mini)
  Demo 2 ─ Complex claim + PII → PII masked, premium model (gpt-4o)
  Demo 3 ─ Same claim again    → instant cache hit, $0 cost
  Demo 4 ─ Primary unavailable → automatic fallback kicks in
  Demo 5 ─ Multi-team load     → seeds the dashboard with team budget data

Run:
    python demo.py
    streamlit run dashboard.py   # open the dashboard afterwards
"""

import os
import random
import time
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

load_dotenv()

from gateway import LLMGateway

console = Console(highlight=False)

# ── Sample claims ─────────────────────────────────────────────────────────────

SIMPLE_CLAIM = """
Claim #AUTO-2024-0512
Vehicle rear-ended at a red light on May 5, 2024. Minor bumper and trunk panel damage.
No injuries reported by either driver. Requesting approval for repair estimate of $780.
""".strip()

COMPLEX_CLAIM = """
Claim #HEALTH-2024-0891
Patient: Sarah Johnson | DOB: 07/22/1985 | SSN: 547-82-3901
Contact: sarah.johnson@gmail.com | Phone: 312-555-8847

Multi-vehicle collision on I-294 northbound on April 12, 2024.
Claimant sustained cervical fracture (C4-C5) requiring surgical intervention
(anterior cervical discectomy and fusion — ACDF). Total hospitalisation: 11 days.

Third-party liability is disputed. Attorney representation confirmed (Chen & Park LLC).
Pre-existing degenerative disc condition documented.
Total medical invoices: $128,450. Lost wages claim: $24,000.
Subrogation against at-fault driver's insurer (Policy #TXL-882-09-Z) initiated.
""".strip()

FRAUD_CLAIM = """
Claim #FRAUD-2024-0103
Claimant IP: 192.168.45.12. This is the third claim filed this quarter.
Damage report inconsistent with collision photos submitted.
Two witness statements contradict the timeline provided in the claim form.
Requesting full investigation before any payout is authorised.
""".strip()

MULTI_TEAM_CLAIMS = [
    ("claims-auto",    "adj_001", SIMPLE_CLAIM),
    ("claims-auto",    "adj_002", "Claim #AUTO-2024-0601. Side-swipe in car park. Paint scratch, no injuries. Estimate $320."),
    ("claims-health",  "adj_003", COMPLEX_CLAIM),
    ("claims-health",  "adj_004", "Patient DOB: 03/15/1978, SSN: 321-54-9876. Slip and fall at workplace. Knee ligament tear, surgery pending. Medical estimate $45,000. Attorney retained."),
    ("fraud-detection","adj_005", FRAUD_CLAIM),
    ("fraud-detection","adj_006", "Claimant email: suspect@tempmail.io. Fourth claim in 8 months. Photos metadata show date discrepancy of 3 days. Escalate to SIU."),
    ("underwriting",   "adj_007", "Standard renewal review. No active claims. Low-risk profile. Recommend 5% premium reduction."),
    ("underwriting",   "adj_008", "High-value commercial fleet renewal. 12 vehicles. Two at-fault incidents in prior 24 months. Recommend premium increase and sub-limit review."),
    ("claims-auto",    "adj_009", SIMPLE_CLAIM),      # will hit cache
    ("claims-health",  "adj_010", COMPLEX_CLAIM),     # will hit cache
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    console.print()


def _print_input(claim_text: str) -> None:
    console.print(Panel(claim_text, title="[dim]Claim Input[/dim]", border_style="dim", padding=(0, 1)))


def _print_steps(result: dict, pii_count: int = 0) -> None:
    steps = Table.grid(padding=(0, 2))
    steps.add_column(style="bold yellow", no_wrap=True)
    steps.add_column()

    # PII
    if result["pii_masked"]:
        pii_label = f"[red]MASKED[/red] — {', '.join(result['pii_types_found'])} redacted"
    else:
        pii_label = "[green]CLEAN[/green] — no PII detected"
    steps.add_row("① PII Check", pii_label)

    # Classification
    complexity_color = "red" if result["task_type"] == "complex" else "green"
    steps.add_row(
        "② Classification",
        f"[{complexity_color}]{result['task_type'].upper()}[/{complexity_color}]  "
        f"(~{result['estimated_tokens']} tokens)  —  {result['task_reason']}",
    )

    # Cache
    cache_label = "[bold green]HIT ✓[/bold green] — served from cache, $0 cost" \
        if result["cache_hit"] else "[yellow]MISS[/yellow] — calling LLM"
    steps.add_row("③ Cache", cache_label)

    # Routing
    if not result["cache_hit"]:
        fallback_note = f"  [red](primary {result['model_attempted']} unavailable → fallback)[/red]" \
            if result["fallback_used"] else ""
        steps.add_row(
            "④ Routing",
            f"[bold]{result['model_used']}[/bold]  [{result['provider']}]{fallback_note}",
        )

    console.print(steps)


def _print_summary(result: dict) -> None:
    console.print()
    console.print(Panel(
        result["summary"],
        title="[bold green]Claim Summary[/bold green]",
        border_style="green",
        padding=(0, 2),
    ))


def _print_metrics(result: dict) -> None:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold blue", padding=(0, 1))
    t.add_column("Request ID")
    t.add_column("Model Used")
    t.add_column("Provider")
    t.add_column("Tokens In")
    t.add_column("Tokens Out")
    t.add_column("Cost USD")
    t.add_column("Latency")
    t.add_column("Cache")
    t.add_column("Fallback")
    t.add_column("PII Masked")

    cache_badge    = "[bold green]HIT[/bold green]"  if result["cache_hit"]    else "[yellow]MISS[/yellow]"
    fallback_badge = "[bold red]YES[/bold red]"       if result["fallback_used"] else "[green]No[/green]"
    pii_badge      = "[bold red]YES[/bold red]"       if result["pii_masked"]    else "[green]No[/green]"

    t.add_row(
        result["request_id"],
        result["model_used"],
        result["provider"],
        str(result["tokens_in"]),
        str(result["tokens_out"]),
        f"${result['cost_usd']:.6f}",
        f"{result['latency_ms']} ms",
        cache_badge,
        fallback_badge,
        pii_badge,
    )
    console.print(t)


# ── Demo scenarios ────────────────────────────────────────────────────────────

def demo_1_simple(gw: LLMGateway) -> None:
    _section("DEMO 1 — Simple Claim  →  Cheap Model (gpt-4o-mini)")
    console.print("[dim]A straightforward auto claim with low token count.[/dim]")
    _print_input(SIMPLE_CLAIM)

    result = gw.process_claim(SIMPLE_CLAIM, team="claims-auto", user_id="adj_001")

    _print_steps(result)
    _print_summary(result)
    _print_metrics(result)


def demo_2_complex_pii(gw: LLMGateway) -> None:
    _section("DEMO 2 — Complex Claim + PII  →  Mask → Premium Model (gpt-4o)")
    console.print("[dim]Multi-party injury claim containing SSN, email, DOB, and phone.[/dim]")
    _print_input(COMPLEX_CLAIM)

    result = gw.process_claim(COMPLEX_CLAIM, team="claims-health", user_id="adj_003")

    _print_steps(result)

    if result["pii_masked"]:
        console.print()
        console.print(Panel(
            result["masked_input_preview"],
            title="[bold red]What the LLM actually received (PII stripped)[/bold red]",
            border_style="red",
            padding=(0, 1),
        ))

    _print_summary(result)
    _print_metrics(result)


def demo_3_cache_hit(gw: LLMGateway) -> None:
    _section("DEMO 3 — Same Claim Again  →  Cache Hit  ($0 cost)")
    console.print("[dim]Submitting the identical complex claim a second time.[/dim]")
    _print_input(COMPLEX_CLAIM)

    result = gw.process_claim(COMPLEX_CLAIM, team="claims-health", user_id="adj_003")

    _print_steps(result)
    _print_summary(result)
    _print_metrics(result)

    console.print("[bold green]✓  Saved LLM call entirely — zero tokens consumed, zero cost.[/bold green]")


def demo_4_fallback(gw: LLMGateway) -> None:
    _section("DEMO 4 — Primary Model Unavailable  →  Automatic Fallback")
    console.print(
        "[dim]Simulating a gpt-4o outage. The gateway detects the failure "
        "and automatically reroutes to gpt-4o-mini.[/dim]"
    )
    _print_input(COMPLEX_CLAIM[:200] + " [truncated for demo]")

    short_claim = "Multi-vehicle collision. Disputed liability. Attorney involved. High medical exposure."
    result = gw.process_claim(
        short_claim,
        team="claims-health",
        user_id="adj_fallback",
        _demo_force_fallback=True,
    )

    _print_steps(result)
    _print_summary(result)
    _print_metrics(result)

    console.print(
        f"[bold yellow]⚡  Primary model was unavailable — "
        f"fallback to [white]{result['model_used']}[/white] completed seamlessly.[/bold yellow]"
    )


def demo_5_multi_team(gw: LLMGateway) -> None:
    _section("DEMO 5 — Multi-Team Simulation  →  Seeds the Dashboard")
    console.print(
        "[dim]Submitting 10 claims from 4 teams to populate budget & usage data.[/dim]\n"
    )

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", padding=(0, 1))
    t.add_column("#",          width=3)
    t.add_column("Team",       style="cyan")
    t.add_column("User")
    t.add_column("Task",       style="bold")
    t.add_column("Model")
    t.add_column("Cache")
    t.add_column("Cost USD")
    t.add_column("Latency")

    for i, (team, user, claim) in enumerate(MULTI_TEAM_CLAIMS, 1):
        result = gw.process_claim(claim, team=team, user_id=user)
        cache_badge = "[green]HIT[/green]" if result["cache_hit"] else "miss"
        t.add_row(
            str(i), team, user,
            f"[red]{result['task_type']}[/red]" if result["task_type"] == "complex"
                else f"[green]{result['task_type']}[/green]",
            result["model_used"],
            cache_badge,
            f"${result['cost_usd']:.6f}",
            f"{result['latency_ms']} ms",
        )
        console.print(f"  [{i}/10] {team} / {user}", end="\r")

    console.print()
    console.print(t)

    # Print team budget summary
    budgets = gw.logger.team_budgets()
    console.print()
    bt = Table(title="[bold]Team Budget Summary[/bold]", box=box.ROUNDED,
               show_header=True, header_style="bold blue", padding=(0, 1))
    bt.add_column("Team")
    bt.add_column("Requests", justify="right")
    bt.add_column("Total Cost", justify="right")
    bt.add_column("Total Tokens", justify="right")
    bt.add_column("Avg Latency", justify="right")
    bt.add_column("Cache Hits", justify="right")

    for b in budgets:
        bt.add_row(
            b["team"],
            str(b["requests"]),
            f"${b['total_cost']:.5f}",
            f"{b['total_tokens']:,}",
            f"{int(b['avg_latency_ms'] or 0)} ms",
            str(b["cache_hits"]),
        )
    console.print(bt)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    console.print()
    console.print(Panel.fit(
        "[bold white]LLM Gateway Demo[/bold white]\n"
        "[dim]Insurance Claim Processing Pipeline[/dim]\n\n"
        "  PII Masking  →  Task Classification  →  Smart Routing\n"
        "  Fallback     →  Response Logging     →  Team Budgets",
        border_style="bright_blue",
        padding=(1, 4),
    ))

    gw = LLMGateway()
    mode_label = "[yellow]MOCK MODE[/yellow]" if gw.mock_mode else "[green]LIVE MODE[/green]"
    console.print(f"\n  Gateway initialised  {mode_label}\n")

    if gw.mock_mode:
        console.print(
            "  [dim]No API keys detected — running with realistic mock responses.\n"
            "  Add OPENAI_API_KEY / GROQ_API_KEY to .env for live LLM calls.[/dim]\n"
        )

    demo_1_simple(gw)
    demo_2_complex_pii(gw)
    demo_3_cache_hit(gw)
    demo_4_fallback(gw)
    demo_5_multi_team(gw)

    console.print()
    console.print(Rule("[bold green]All demos complete[/bold green]", style="green"))
    console.print()
    console.print(
        "  [bold]Next step:[/bold]  run [cyan]streamlit run dashboard.py[/cyan]  "
        "to see the team budget dashboard.\n"
    )


if __name__ == "__main__":
    main()
