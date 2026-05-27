"""
pps_app.py
==========

Pet Parenting Style (PPS) — Streamlit app: one item per screen, research consent
flow, contact, optional pet demographics, result, completion — aligned with
``DSLQ_App/dslq_app.py``. Scoring: ``pps_scoring_baseline.py`` + margin labels
``pps_mixed_ambiguous_logic`` (ambiguous if margin < 1/24).

    streamlit run pps_app.py

Supabase: tables ``pps_sessions`` and ``pps_contacts`` (see ``pps_research.py``).
"""

from __future__ import annotations

import html
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from pps_research import (
    build_pps_export,
    is_valid_email_simple,
    persist_pps_contact,
    persist_pps_session,
)
from pps_scoring_baseline import (
    ITEM_TEXTS,
    LIKERT_LABELS,
    chart_similarity_donut,
    compute_from_likert_labels,
    margin_and_confidence_from_eff_dists,
    ordered_item_ids,
)

# ── Page + style (shared with DSLQ) ───────────────────────────────────────
st.set_page_config(
    page_title="Pet Parenting Style Questionnaire",
    page_icon="🐾",
    layout="centered",
)

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
    background-color: #F7F8FA;
    color: #1C1C1E;
}
.dslq-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.dslq-section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8E8E93;
    margin-bottom: 6px;
}
.dslq-question {
    font-size: 1.02rem;
    font-weight: 500;
    line-height: 1.6;
    color: #1C1C1E;
}
.band-style {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.95rem;
}
.style-authoritarian { background:#FEE2E2; color:#991B1B; }
.style-authoritative { background:#D1FAE5; color:#065F46; }
.style-permissive    { background:#FEF9C3; color:#854D0E; }
.band-mixed { background:#EDE9FE; color:#5B21B6; padding:6px 16px; border-radius:20px;
              font-weight:700; font-size:0.95rem; display:inline-block; }
.intro-lead {
    font-size: 1.02rem;
    font-weight: 400;
    line-height: 1.65;
    color: #1C1C1E;
    margin: 0 0 14px 0;
    white-space: pre-line;
}
.intro-card-body {
    font-size: 1.02rem;
    font-weight: 400;
    line-height: 1.6;
    color: #1C1C1E;
    margin: 0;
}
.intro-card-body + .intro-card-body { margin-top: 12px; }
.pps-result-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #374151;
    margin-bottom: 6px;
}
.result-graph-title {
    font-size: 1.08rem;
    font-weight: 600;
    line-height: 1.4;
    color: #1C1C1E;
    margin: 0 0 6px 0;
}
.result-graph-caption {
    font-size: 1.0rem;
    font-weight: 400;
    line-height: 1.55;
    color: #1C1C1E;
    margin: 0 0 14px 0;
}
.result-disclaimer {
    font-size: 0.9rem;
    line-height: 1.5;
    color: #1C1C1E;
    margin-top: 16px;
}
.stButton > button[kind="primary"],
div[data-testid="stBaseButton-primary"] button,
div[data-testid="stBaseButton-primary"] > button {
    background-color: #FF6B5B !important;
    border-color: #FF6B5B !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover,
div[data-testid="stBaseButton-primary"] button:hover,
div[data-testid="stBaseButton-primary"] > button:hover {
    background-color: #E85A4A !important;
    border-color: #E85A4A !important;
    color: #FFFFFF !important;
}
.disclaimer { font-size:0.78rem; color:#8E8E93; line-height:1.5; margin-top:12px; }
.progress-label { font-size:0.8rem; color:#8E8E93; margin-bottom:2px; }
</style>
""",
    unsafe_allow_html=True,
)

_ROOT = Path(__file__).resolve().parent
_COPY_CANDIDATES = [_ROOT, _ROOT.parent, Path.cwd()]


def _read_csv(path: Path) -> pd.DataFrame:
    last_err: Exception = RuntimeError("No encodings tried")
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


@st.cache_data(show_spinner=False)
def load_copy_csv() -> Dict[str, str]:
    filename = "PPS_App_Copy_Extended.csv"
    for base in _COPY_CANDIDATES:
        p = base / filename
        if p.is_file():
            df = _read_csv(p)
            out: Dict[str, str] = {}
            for _, row in df.iterrows():
                key = str(row["key"]).strip()
                text = str(row["text"]) if pd.notna(row["text"]) else ""
                text = text.replace("\ufffd", "'").replace("\u2019", "'")
                out[key] = text
            return out
    raise FileNotFoundError(f"Place {filename} next to pps_app.py or run from repo root.")


@st.cache_data(show_spinner=False)
def load_optional_modules_df() -> pd.DataFrame:
    p = _ROOT / "PPS_App_OptionalModules.csv"
    for base in _COPY_CANDIDATES:
        q = base / "PPS_App_OptionalModules.csv"
        if q.is_file():
            return _read_csv(q)
    return pd.DataFrame()


COPY = load_copy_csv()
OPTIONAL_DF = load_optional_modules_df()


def c(key: str, default: str = "") -> str:
    val = COPY.get(key, "")
    return val if val and str(val).strip() else default


def style_band_class(style: str) -> str:
    if style == "Authoritarian":
        return "style-authoritarian"
    if style == "Authoritative":
        return "style-authoritative"
    return "style-permissive"


def interpretation_key_for_style(style: str) -> str:
    return {
        "Authoritative": "result_authoritative",
        "Authoritarian": "result_authoritarian",
        "Permissive": "result_permissive",
    }[style]


def strip_style_headline_for_mixed_body(full_interp: str) -> str:
    text = full_interp.strip()
    if not text.lower().startswith("your parenting style:"):
        return text
    parts = text.split("\n", 2)
    if len(parts) >= 3:
        return parts[2].lstrip("\n").strip()
    return "\n".join(parts[1:]).strip() if len(parts) > 1 else text


def mixed_secondary_snippet(secondary_style: str) -> str:
    key = f"result_mixed_secondary_{secondary_style.lower()}"
    return c(key, "")


def _normalize_mixed_markdown(text: str) -> str:
    text = re.sub(
        r"(which is characterized by:)\s*\n+",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_mixed_intro_text(primary_style: str, secondary_style: str) -> str:
    tmpl = c("result_mixed_intro_template", "")
    primary_desc = c(interpretation_key_for_style(primary_style), "").strip()
    secondary_desc = c(interpretation_key_for_style(secondary_style), "").strip()
    filled = tmpl.format(
        primary_style=primary_style,
        secondary_style=secondary_style,
        primary_description=primary_desc,
        secondary_description=secondary_desc,
    )
    return _normalize_mixed_markdown(filled)


def render_result_body(computed) -> None:
    margin, conf_label, primary_style, secondary_style = margin_and_confidence_from_eff_dists(
        computed.eff_dists
    )
    st.markdown(f"## {c('result_title', 'Your result')}")

    if conf_label == "Ambiguous (Borderline)":
        st.markdown(
            f'<p><span class="band-mixed">{html.escape(c("result_mixed_label", "Mixed style"))}</span></p>',
            unsafe_allow_html=True,
        )
        st.markdown(build_mixed_intro_text(primary_style, secondary_style))
    else:
        band_cls = style_band_class(primary_style)
        st.markdown(
            f'<p class="pps-result-label">{html.escape(c("result_style_prefix", "Your parenting style"))}</p>'
            f'<p><span class="band-style {band_cls}">{html.escape(primary_style)}</span></p>',
            unsafe_allow_html=True,
        )
        st.markdown(c(interpretation_key_for_style(primary_style)))
        if conf_label == "Moderate Confidence":
            st.markdown("---")
            overlap = mixed_secondary_snippet(secondary_style)
            clearer = c("result_clearer_label", "Overlap between styles")
            st.markdown(f"**{clearer}**")
            mod_body = c(
                "result_moderate_overlap_body",
                "In addition to the **{primary_style}** pattern above, your answers "
                "also show meaningful overlap with the **{secondary_style}** style: {overlap}",
            )
            st.markdown(
                mod_body.format(
                    primary_style=primary_style,
                    secondary_style=secondary_style,
                    overlap=overlap,
                )
            )

    st.markdown("---")
    st.markdown(
        f'<p class="result-graph-title">{html.escape(c("result_graph_title", "Your profile"))}</p>'
        f'<p class="result-graph-caption">{html.escape(c("result_graph_explainer", ""))}</p>',
        unsafe_allow_html=True,
    )
    st.altair_chart(chart_similarity_donut(computed), use_container_width=True)
    st.markdown(
        f'<p class="result-disclaimer">{html.escape(c("result_disclaimer", ""))}</p>',
        unsafe_allow_html=True,
    )


def render_supabase_diagnostics() -> None:
    err = st.session_state.get("supabase_error")
    tb = st.session_state.get("supabase_traceback")
    if err or tb:
        st.markdown("---")
        st.markdown("### Supabase (sessions)")
        if err:
            st.error(err)
        if tb:
            st.code(tb)
    c_err = st.session_state.get("supabase_contact_error")
    c_tb = st.session_state.get("supabase_contact_traceback")
    if c_err or c_tb:
        st.markdown("---")
        st.markdown("### Supabase (contacts)")
        if c_err:
            st.error(c_err)
        if c_tb:
            st.code(c_tb)


ITEM_IDS = ordered_item_ids()
N_ITEMS = len(ITEM_IDS)


def _reset_research_state() -> None:
    st.session_state["choices"] = {
        "share_questionnaire_data": False,
        "future_contact": False,
    }
    st.session_state["dog_demo"] = {}
    st.session_state["contact"] = {}
    st.session_state["export_blob"] = None
    st.session_state["saved_path"] = None
    for k in (
        "supabase_error",
        "supabase_traceback",
        "supabase_insert_ok",
        "supabase_contact_error",
        "supabase_contact_traceback",
        "supabase_contact_insert_ok",
    ):
        st.session_state[k] = None


def init_state() -> None:
    defaults: Dict[str, Any] = {
        "page": "intro",
        "item_index": 0,
        "responses": {},
        "score_result": None,
        "computed": None,
        "session_id": str(uuid.uuid4()),
    }
    defaults["choices"] = {
        "share_questionnaire_data": False,
        "future_contact": False,
    }
    defaults["dog_demo"] = {}
    defaults["contact"] = {}
    defaults["export_blob"] = None
    defaults["saved_path"] = None
    for k in (
        "supabase_error",
        "supabase_traceback",
        "supabase_insert_ok",
        "supabase_contact_error",
        "supabase_contact_traceback",
        "supabase_contact_insert_ok",
    ):
        defaults[k] = None
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def go(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


def wrap_up_and_show_result() -> None:
    computed = st.session_state.get("score_result")
    if computed is None:
        computed = compute_from_likert_labels(
            {k: st.session_state["responses"].get(k) for k in ITEM_IDS}
        )
        st.session_state["score_result"] = computed
    if computed is None:
        st.error("Cannot compute score. Please complete the questionnaire again.")
        st.stop()
    m, cl, p_st, s_st = margin_and_confidence_from_eff_dists(computed.eff_dists)
    ch = st.session_state["choices"]
    payload = build_pps_export(
        computed,
        m,
        cl,
        p_st,
        s_st,
        {k: st.session_state["responses"].get(k) for k in ITEM_IDS},
        st.session_state["dog_demo"],
        st.session_state["contact"],
        ch,
    )
    st.session_state["export_blob"] = payload
    if ch.get("share_questionnaire_data"):
        pth = persist_pps_session(payload, _ROOT)
        st.session_state["saved_path"] = str(pth) if pth else None
    else:
        st.session_state["saved_path"] = None
    persist_pps_contact(payload)
    st.session_state["computed"] = computed
    go("result")


# ── Screens ────────────────────────────────────────────────────────────────


def screen_intro() -> None:
    st.markdown(f"## {c('app_title', 'Pet Parenting Style Questionnaire')}")
    st.markdown(
        f'<p class="intro-lead">{html.escape(c("intro_body"))}</p>',
        unsafe_allow_html=True,
    )
    st.info(
        f"⏱ {c('intro_time_estimate', 'Most people complete this questionnaire in about 5–10 minutes.')}"
    )
    st.markdown(
        '<div class="dslq-card">'
        f'<p class="intro-card-body">{html.escape(c("intro_scale_explainer"))}</p>'
        f'<p class="intro-card-body">{html.escape(c("intro_note"))}</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button(c("start_button", "Start questionnaire →"), type="primary", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["item_index"] = 0
        st.session_state["responses"] = {}
        st.session_state["score_result"] = None
        st.session_state["computed"] = None
        _reset_research_state()
        go("questionnaire")


def screen_questionnaire() -> None:
    idx = int(st.session_state["item_index"])
    if idx >= N_ITEMS:
        go("sharing")
        return

    item_id = ITEM_IDS[idx]
    st.progress((idx + 1) / N_ITEMS)
    st.markdown(
        f'<p class="progress-label">Question {idx + 1} of {N_ITEMS}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="dslq-card">'
        f'<p class="dslq-section-label">Question {idx + 1}</p>'
        f'<p class="dslq-question">{ITEM_TEXTS[item_id]}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    likert_display = [
        c("likert_1", LIKERT_LABELS[0]),
        c("likert_2", LIKERT_LABELS[1]),
        c("likert_3", LIKERT_LABELS[2]),
        c("likert_4", LIKERT_LABELS[3]),
        c("likert_5", LIKERT_LABELS[4]),
    ]
    label_to_canonical = {disp: LIKERT_LABELS[i] for i, disp in enumerate(likert_display)}

    current = st.session_state["responses"].get(item_id)
    default_display = None
    if current is not None:
        for disp, canon in label_to_canonical.items():
            if canon == current:
                default_display = disp
                break

    choice_display = st.radio(
        "Select one:",
        options=likert_display,
        index=likert_display.index(default_display) if default_display in likert_display else None,
        horizontal=False,
        key=f"likert_{item_id}",
    )

    st.markdown("")
    col_back, col_next = st.columns([1, 2])
    is_last = idx == N_ITEMS - 1
    back_label = c("back_button", "← Back")
    next_label = c("continue_button", "Continue →")
    back = col_back.button(back_label, disabled=(idx == 0), key=f"back_{idx}")
    nxt = col_next.button(next_label, type="primary", key=f"next_{idx}")

    if back:
        st.session_state["item_index"] = max(0, idx - 1)
        st.rerun()

    if nxt:
        if choice_display is None:
            st.warning(c("complete_all_items_warning", "Please answer before continuing."))
            st.stop()
        canonical = label_to_canonical[choice_display]
        st.session_state["responses"][item_id] = canonical
        if is_last:
            comp = compute_from_likert_labels(
                {k: st.session_state["responses"].get(k) for k in ITEM_IDS}
            )
            if comp is None:
                st.error(c("complete_all_items_warning", "Please answer all questions."))
                st.stop()
            st.session_state["score_result"] = comp
            st.session_state["computed"] = comp
            go("sharing")
        else:
            st.session_state["item_index"] = idx + 1
            st.rerun()


def screen_sharing() -> None:
    st.markdown(f"## {c('save_data_title', 'Share data?')}")
    st.markdown(
        c(
            "save_data_body",
            "Sharing your data is optional. You can receive your result whether or not you choose to share anything.",
        )
    )
    st.markdown(
        '<div class="dslq-card"><b>'
        + html.escape(c("consent_card_title", "Research consent"))
        + "</b><br><br>"
        + html.escape(
            c(
                "consent_card_body",
                "If you opt in, your data may be used for research only.",
            )
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    with st.form("pps_sharing_form"):
        share_q = st.checkbox(
            c(
                "share_q_checkbox",
                "I agree to share my questionnaire data for research.",
            ),
            value=bool(st.session_state["choices"].get("share_questionnaire_data")),
        )
        c_back, c_cont = st.columns([1, 2])
        b_back = c_back.form_submit_button(c("back_button", "← Back"))
        b_cont = c_cont.form_submit_button(c("continue_button", "Continue →"), type="primary")

    if b_back:
        st.session_state["item_index"] = max(0, N_ITEMS - 1)
        go("questionnaire")
    if b_cont:
        st.session_state["choices"]["share_questionnaire_data"] = share_q
        go("contact")


def screen_contact() -> None:
    contact: Dict[str, Any] = st.session_state.get("contact", {}) or {}
    ch = st.session_state["choices"]

    st.markdown(f"## {c('contact_screen_title', 'Stay in touch')}")
    st.markdown(
        c(
            "contact_screen_body",
            "Would you like to be notified about future study opportunities?",
        )
    )
    want_raw = contact.get("future_contact", "No")
    idx = 1 if want_raw in (True, "Yes", "yes") else 0
    wants = st.radio(
        "Select one:",
        options=["No", "Yes"],
        index=idx,
        horizontal=True,
        key="pps_contact_wants",
    )
    name_val = str(contact.get("contact_name", ""))
    email_val = str(contact.get("contact_email", ""))
    if wants == "Yes":
        st.markdown("---")
        name_val = st.text_input(
            c("contact_name_label", "Your name (first and last)"),
            value=name_val,
            key="pps_cname",
        )
        email_val = st.text_input(
            c("contact_email_label", "Your email address"),
            value=email_val,
            key="pps_email",
        )

    st.markdown("")
    col_b, col_n = st.columns([1, 2])
    b_back = col_b.button(c("back_button", "← Back"), key="pps_con_back")
    b_next = col_n.button(c("continue_button", "Continue →"), type="primary", key="pps_con_next")

    if b_back:
        go("sharing")
    if b_next:
        if wants == "Yes":
            if not str(email_val or "").strip():
                st.error("Please enter an email address, or choose No for future contact.")
                st.stop()
            if not is_valid_email_simple(email_val):
                st.error("Please enter a valid email address (e.g. you@example.com).")
                st.stop()
        st.session_state["choices"]["future_contact"] = wants == "Yes"
        contact["future_contact"] = wants
        contact["contact_name"] = name_val
        contact["contact_email"] = email_val
        st.session_state["contact"] = contact
        if ch.get("share_questionnaire_data"):
            go("demographics")
        else:
            wrap_up_and_show_result()


def screen_demographics() -> None:
    ch = st.session_state["choices"]
    share_q = ch["share_questionnaire_data"]
    dog_demo: Dict[str, Any] = dict(st.session_state.get("dog_demo", {}))
    contact: Dict[str, Any] = dict(st.session_state.get("contact", {}))

    if OPTIONAL_DF.empty or not share_q:
        wrap_up_and_show_result()
        return

    with st.form("pps_demo_form"):
        if share_q:
            st.markdown(f"### {c('pet_demo_title', 'Optional pet demographics')}")
            drows = OPTIONAL_DF[OPTIONAL_DF["section"] == "dog_demographics_optional"].sort_values(
                "display_order"
            )
            for _, f in drows.iterrows():
                fk = str(f["field_key"])
                qtxt = str(f["question_text"])
                rtype = str(f["response_type"])
                if rtype == "text":
                    dog_demo[fk] = st.text_input(qtxt, value=str(dog_demo.get(fk, "")), key=f"dd_{fk}")
                elif rtype == "number":
                    dog_demo[fk] = st.number_input(
                        qtxt,
                        min_value=0,
                        step=1,
                        value=int(dog_demo.get(fk, 0) or 0),
                        key=f"dd_{fk}",
                    )
                elif rtype == "single_select":
                    opts = [o.strip() for o in str(f["options_pipe_delimited"]).split("|")]
                    cur = dog_demo.get(fk, opts[0])
                    dog_demo[fk] = st.selectbox(
                        qtxt,
                        options=opts,
                        index=opts.index(cur) if cur in opts else 0,
                        key=f"dd_{fk}",
                    )
                elif rtype == "single_select_plus_text":
                    opts = [o.strip() for o in str(f["options_pipe_delimited"]).split("|")]
                    cur = dog_demo.get(fk, opts[0])
                    sel = st.selectbox(
                        qtxt,
                        options=opts,
                        index=opts.index(cur) if cur in opts else 0,
                        key=f"dd_{fk}_s",
                    )
                    extra = st.text_input(
                        "Details (if applicable):",
                        value=str(dog_demo.get(f"{fk}_txt", "")),
                        key=f"dd_{fk}_t",
                    )
                    dog_demo[fk] = sel
                    dog_demo[f"{fk}_txt"] = extra

        c_b, c_f = st.columns([1, 2])
        b_b = c_b.form_submit_button(c("back_button", "← Back"))
        b_f = c_f.form_submit_button(c("continue_button", "Continue →"), type="primary")

    if b_b:
        go("contact")
    if b_f:
        st.session_state["dog_demo"] = dog_demo
        st.session_state["contact"] = contact
        st.session_state["choices"] = ch
        wrap_up_and_show_result()


def screen_result() -> None:
    computed = st.session_state.get("score_result")
    if computed is None:
        computed = compute_from_likert_labels(
            {k: st.session_state["responses"].get(k) for k in ITEM_IDS}
        )
        st.session_state["score_result"] = computed
    if computed is None:
        st.error(c("complete_all_items_warning", "Please complete the questionnaire first."))
        if st.button(c("back_button", "← Back")):
            go("questionnaire")
        return

    render_result_body(computed)
    render_supabase_diagnostics()

    col_b, col_n = st.columns([1, 2])
    if col_b.button(c("back_button", "← Back"), key="result_back"):
        ch = st.session_state.get("choices", {})
        if ch.get("share_questionnaire_data"):
            go("demographics")
        else:
            go("contact")
    if col_n.button(c("result_continue", "Continue →"), type="primary", key="res_next"):
        go("completion")


def screen_completion() -> None:
    ch = st.session_state.get("choices", {})
    saved_data = bool(ch.get("share_questionnaire_data"))
    wants_contact = bool(
        ch.get("future_contact")
        and str((st.session_state.get("contact") or {}).get("contact_email") or "").strip()
    )
    contact_saved = st.session_state.get("supabase_contact_insert_ok") is True
    st.markdown(f"## {c('completion_thank', 'Thank you!')}")
    if saved_data and contact_saved:
        st.success(c("completion_saved_both"))
    elif (not saved_data) and contact_saved and wants_contact:
        st.success(c("completion_contact_only"))
    elif saved_data:
        st.success(c("completion_saved_research"))
    else:
        st.info(c("completion_no_save"))
    render_supabase_diagnostics()
    st.markdown("---")
    if st.button("Start over", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


PAGES: Dict[str, Any] = {
    "intro": screen_intro,
    "questionnaire": screen_questionnaire,
    "sharing": screen_sharing,
    "contact": screen_contact,
    "demographics": screen_demographics,
    "result": screen_result,
    "completion": screen_completion,
}

page = st.session_state.get("page", "intro")
PAGES.get(page, screen_intro)()
