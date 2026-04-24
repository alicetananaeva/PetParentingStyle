#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPS research export + storage (Parallels :mod:`dslq_app` ``build_export`` / Supabase insert).

- Optional local JSON if ``STORAGE_MODE == "local"``.
- ``supabase`` mode posts to ``pps_sessions`` and ``pps_contacts`` (create matching tables in Supabase).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from pps_scoring_baseline import (
    LIKERT_LABELS,
    PPSComputed,
    likert_label_to_score,
    ordered_item_ids,
)
from pps_mixed_ambiguous_logic import ConfidenceLabel

STORAGE_MODE = "supabase"  # "none" | "local" | "supabase"
APP_VERSION = "v1"
APP_TABLE_SESSIONS = "pps_sessions"
APP_TABLE_CONTACTS = "pps_contacts"

# Identity / contact must not be duplicated in human_demographics (same as DSLQ).
HUMAN_DEMO_EXCLUDED_FIELD_KEYS = frozenset(
    {
        "contact_name",
        "contact_email",
        "first_name",
        "last_name",
        "surname",
        "human_first_name",
        "human_last_name",
        "participant_name",
        "owner_name",
        "human_name",
        "future_contact",
    }
)


def _human_demo_for_export(human_demo: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in human_demo.items() if k not in HUMAN_DEMO_EXCLUDED_FIELD_KEYS}


def answers_to_item_numeric(responses: Dict[str, Optional[str]]) -> Dict[str, int]:
    """Map item id -> Likert 1-5; raises if missing/invalid."""
    out: Dict[str, int] = {}
    for iid in ordered_item_ids():
        lbl = responses.get(iid)
        if lbl is None:
            raise ValueError(f"Missing response for {iid}")
        out[iid] = likert_label_to_score(str(lbl))
    return out


def _set_supabase_diag(error: Optional[str], traceback_text: Optional[str]) -> None:
    if error:
        st.session_state["supabase_error"] = str(error)
    else:
        st.session_state["supabase_error"] = None
    if traceback_text:
        st.session_state["supabase_traceback"] = str(traceback_text)
    else:
        st.session_state["supabase_traceback"] = None


def _set_supabase_contact_diag(error: Optional[str], traceback_text: Optional[str]) -> None:
    if error:
        st.session_state["supabase_contact_error"] = str(error)
    else:
        st.session_state["supabase_contact_error"] = None
    if traceback_text:
        st.session_state["supabase_contact_traceback"] = str(traceback_text)
    else:
        st.session_state["supabase_contact_traceback"] = None


def is_valid_email_simple(email: str) -> bool:
    """Practical check — same pattern as ``dslq_app``."""
    e = str(email or "")
    if not e or e.strip() != e:
        return False
    if " " in e:
        return False
    if "@" not in e:
        return False
    local, domain = e.split("@", 1)
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    if domain.startswith(".") or domain.endswith(".") or ".." in domain:
        return False
    return True


def build_pps_export(
    computed: PPSComputed,
    margin: float,
    conf_label: ConfidenceLabel,
    primary_style: str,
    secondary_style: str,
    responses: Dict[str, Optional[str]],
    dog_demo: Dict[str, Any],
    human_demo: Dict[str, Any],
    contact: Dict[str, Any],
    choices: Dict[str, Any],
) -> Dict[str, Any]:
    item_numeric = answers_to_item_numeric({k: responses.get(k) for k in ordered_item_ids()})

    return {
        "session_id": st.session_state.get("session_id", str(uuid.uuid4())),
        "app_version": APP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parenting_style": computed.final_style,
        "parenting_style_display_primary": primary_style,
        "parenting_style_display_secondary": secondary_style,
        "dist_margin": margin,
        "confidence_label": conf_label,
        "mean_permissive": float(computed.s_perm),
        "mean_authoritative": float(computed.s_authv),
        "mean_authoritarian": float(computed.s_authn),
        "z_permissive": float(computed.z_perm),
        "z_authoritative": float(computed.z_authv),
        "z_authoritarian": float(computed.z_authn),
        "p_permissive": float(computed.p_perm),
        "p_authoritative": float(computed.p_authv),
        "p_authoritarian": float(computed.p_authn),
        "eff_dists": dict(computed.eff_dists),
        "sim_scores": dict(computed.sim_scores),
        "item_responses_1_5": item_numeric,
        "item_responses_labels": {k: responses[k] for k in ordered_item_ids() if k in responses},
        "raw_answers_json": {"pps_items": item_numeric, "likert_order": list(LIKERT_LABELS)},
        "research_choices": dict(choices),
        "dog_demographics": dog_demo,
        "human_demographics": _human_demo_for_export(human_demo),
        "contact_details": contact,
    }


def _supabase_pps_insert(payload: Dict[str, Any]) -> bool:
    try:
        import urllib.error
        import urllib.request

        secrets = st.secrets["supabase"]
        url = secrets["url"].rstrip("/") + f"/rest/v1/{APP_TABLE_SESSIONS}"
        key = secrets["key"]
        ch = payload.get("research_choices") or {}
        record: Dict[str, Any] = {
            "session_id": payload.get("session_id"),
            "app_version": payload.get("app_version"),
            "parenting_style": payload.get("parenting_style"),
            "dist_margin": payload.get("dist_margin"),
            "confidence_label": payload.get("confidence_label"),
            "mean_permissive": payload.get("mean_permissive"),
            "mean_authoritative": payload.get("mean_authoritative"),
            "mean_authoritarian": payload.get("mean_authoritarian"),
            "z_permissive": payload.get("z_permissive"),
            "z_authoritative": payload.get("z_authoritative"),
            "z_authoritarian": payload.get("z_authoritarian"),
            "p_permissive": payload.get("p_permissive"),
            "p_authoritative": payload.get("p_authoritative"),
            "p_authoritarian": payload.get("p_authoritarian"),
            "eff_dists": payload.get("eff_dists"),
            "sim_scores": payload.get("sim_scores"),
            "item_responses_1_5": payload.get("item_responses_1_5"),
            "raw_answers_json": payload.get("raw_answers_json"),
            "research_choices": payload.get("research_choices"),
            "dog_demographics": payload.get("dog_demographics"),
            "human_demographics": payload.get("human_demographics"),
            "consented_questionnaire": bool(ch.get("share_questionnaire_data")),
            "consented_demographics": bool(ch.get("share_demographic_data")),
        }
        # Drop None values for optional columns
        record = {k: v for k, v in record.items() if v is not None}
        data = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Prefer": "return=minimal",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status in (200, 201)
            if ok:
                _set_supabase_diag(None, None)
            else:
                _set_supabase_diag(
                    f"PPS Supabase insert failed (HTTP {resp.status}).",
                    None,
                )
            return ok
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_bytes = e.read()
            body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
        except Exception:
            body_text = ""
        detail = body_text.strip() if body_text.strip() else "(empty response body)"
        _set_supabase_diag(
            f"PPS Supabase insert failed: HTTP {e.code} {e.reason}. Response: {detail}",
            None,
        )
        return False
    except Exception as e:
        import traceback

        _set_supabase_diag(
            f"PPS Supabase insert failed: {e}",
            traceback.format_exc(),
        )
        return False


def _supabase_pps_contact_insert(payload: Dict[str, Any]) -> bool:
    try:
        import urllib.error
        import urllib.request

        secrets = st.secrets["supabase"]
        url = secrets["url"].rstrip("/") + f"/rest/v1/{APP_TABLE_CONTACTS}"
        key = secrets["key"]
        contact = payload.get("contact_details") or {}
        record = {
            "session_id": payload.get("session_id"),
            "created_at": payload.get("created_at"),
            "name": contact.get("contact_name") or None,
            "email": contact.get("contact_email") or None,
            "consented_future_contact": bool(
                (payload.get("research_choices") or {}).get("future_contact")
            ),
            "app_version": payload.get("app_version"),
        }
        data = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Prefer": "return=minimal",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status in (200, 201)
            if ok:
                _set_supabase_contact_diag(None, None)
            else:
                _set_supabase_contact_diag(
                    f"PPS contact insert failed (HTTP {resp.status}).",
                    None,
                )
            return ok
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_bytes = e.read()
            body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
        except Exception:
            body_text = ""
        detail = body_text.strip() if body_text.strip() else "(empty response body)"
        _set_supabase_contact_diag(
            f"PPS contact insert failed: HTTP {e.code} {e.reason}. Response: {detail}",
            None,
        )
        return False
    except Exception as e:
        import traceback

        _set_supabase_contact_diag(
            f"PPS contact insert failed: {e}",
            traceback.format_exc(),
        )
        return False


def persist_pps_session(payload: Dict[str, Any], base_dir: Path) -> Optional[Path]:
    if STORAGE_MODE == "none":
        return None
    if STORAGE_MODE == "supabase":
        st.session_state["supabase_insert_ok"] = _supabase_pps_insert(payload)
        return None
    try:
        out_dir = base_dir / "pps_sessions"
        out_dir.mkdir(exist_ok=True, parents=True)
        out = out_dir / f"{payload['session_id']}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    except Exception:
        return None


def persist_pps_contact(payload: Dict[str, Any]) -> None:
    if STORAGE_MODE != "supabase":
        st.session_state["supabase_contact_insert_ok"] = None
        _set_supabase_contact_diag(None, None)
        return
    contact = payload.get("contact_details") or {}
    wants = bool((payload.get("research_choices") or {}).get("future_contact"))
    email = str(contact.get("contact_email") or "").strip()
    if wants and email:
        st.session_state["supabase_contact_insert_ok"] = _supabase_pps_contact_insert(payload)
    else:
        st.session_state["supabase_contact_insert_ok"] = None
        _set_supabase_contact_diag(None, None)
