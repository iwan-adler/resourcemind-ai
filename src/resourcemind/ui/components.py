"""Small presentation helpers for the Streamlit app (styling only, no computation)."""

from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
:root {
  --rm-card: #171A1F; --rm-border: #2A2F37; --rm-muted: #9CA3AF;
  --rm-red: #F87171; --rm-amber: #FBBF24; --rm-green: #4ADE80; --rm-blue: #7DD3FC;
}
.block-container { padding-top: 2.2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.rm-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px;
  margin: 0.4rem 0 1rem 0; }
.rm-card { background: var(--rm-card); border: 1px solid var(--rm-border); border-radius: 8px;
  padding: 12px 14px; border-left: 4px solid var(--rm-border); }
.rm-card.tone-critical { border-left-color: var(--rm-red); }
.rm-card.tone-warning { border-left-color: var(--rm-amber); }
.rm-card.tone-good { border-left-color: var(--rm-green); }
.rm-card.tone-info { border-left-color: var(--rm-blue); }
.rm-label { color: var(--rm-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
.rm-value { font-size: 1.65rem; font-weight: 650; line-height: 1.25; margin-top: 2px; color: #F3F4F6; }
.rm-sub { color: var(--rm-muted); font-size: 0.82rem; margin-top: 2px; }
.rm-banner { border: 1px solid #3F3A2A; background: #1F1B12; color: #FDE68A; border-radius: 6px;
  padding: 8px 12px; font-size: 0.86rem; margin-bottom: 0.8rem; }
.rm-pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
  font-weight: 600; border: 1px solid currentColor; margin-right: 6px; }
.rm-pill.critical { color: var(--rm-red); } .rm-pill.warning { color: var(--rm-amber); }
.rm-pill.good { color: var(--rm-green); } .rm-pill.info { color: var(--rm-blue); }
</style>
"""

TONE_WORD = {"critical": "Alert", "warning": "Caution", "good": "OK", "info": "Info", "neutral": ""}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(
        '<div class="rm-banner" role="note"><strong>Decision-support prototype.</strong> '
        "All data are synthetic. Recommendations are model outputs for human review, "
        "not operational dispatch instructions.</div>",
        unsafe_allow_html=True,
    )


def cards(items: list[dict]) -> None:
    """items: dicts with label, value, optional sub and tone (critical/warning/good/info/neutral)."""
    parts = []
    for it in items:
        tone = it.get("tone", "neutral")
        word = TONE_WORD.get(tone, "")
        aria = html.escape(f"{it['label']}: {it['value']}. {it.get('sub', '')} {word}".strip())
        sub = f'<div class="rm-sub">{html.escape(str(it["sub"]))}</div>' if it.get("sub") else ""
        parts.append(
            f'<div class="rm-card tone-{tone}" role="group" aria-label="{aria}">'
            f'<div class="rm-label">{html.escape(it["label"])}</div>'
            f'<div class="rm-value">{html.escape(str(it["value"]))}</div>{sub}</div>'
        )
    st.markdown(f'<div class="rm-cards">{"".join(parts)}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "info") -> str:
    return f'<span class="rm-pill {tone}">{html.escape(text)}</span>'


def status_tone(status: str) -> str:
    return {
        "OPTIMAL": "good",
        "FEASIBLE": "warning",
        "INFEASIBLE": "critical",
        "UNKNOWN": "critical",
        "MODEL_INVALID": "critical",
        "INVALID": "critical",
    }.get(status, "info")
