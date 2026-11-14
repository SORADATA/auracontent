"""Reprise : sujet, titre et accroche validés, gardés tant que le script n'a pas abouti.

Choisir un sujet coûte des appels LLM (page, titre, retournement, doublon, accroches). Si l'écriture du
script échoue (quotas), l'exécution suivante reprend ce sujet au lieu de repartir de zéro.

Fichier : assets/shared/state/pending_topics.json, une entrée par chaîne, synchronisé sur Hugging Face
(y compris quand la génération échoue, voir cli/generate.py).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from auracontent.services.publishing.state_sync import STATE_DIR

PENDING_FILE = "pending_topics.json"
PENDING_PATH = STATE_DIR / PENDING_FILE


def _path(channel: str | None = None):
    """Chaîne à état isolé (`state.isolated`) : son propre fichier ; sinon le fichier partagé historique."""
    from auracontent.services.publishing.state_sync import channel_state_dir

    own = channel_state_dir(channel)
    return own / PENDING_FILE if own else PENDING_PATH


def _load(channel: str | None = None) -> dict:
    try:
        data = json.loads(_path(channel).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data: dict, channel: str | None = None) -> None:
    path = _path(channel)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        print(f"⚠️ Sujet en attente non enregistré : {e}")


def save_pending(channel: str, state: dict) -> None:
    """Enregistre (ou met à jour) l'état validé de la chaîne ; compte les tentatives d'écriture du script."""
    data = _load(channel)
    previous = data.get(channel) or {}
    same = previous.get("topic") == state.get("topic")
    data[channel] = {**state, "saved_at": previous.get("saved_at") if same else
                     datetime.now(UTC).isoformat(timespec="seconds"),
                     "script_attempts": int(previous.get("script_attempts", 0)) if same else 0}
    _write(data, channel)


def note_script_attempt(channel: str) -> int:
    """Compte un échec du script sur le sujet en attente ; renvoie le nombre d'échecs (0 si aucun sujet)."""
    data = _load(channel)
    if channel not in data:
        return 0
    data[channel]["script_attempts"] = int(data[channel].get("script_attempts", 0)) + 1
    _write(data, channel)
    return data[channel]["script_attempts"]


def load_pending(channel: str, max_attempts: int = 3, max_age_days: float = 7) -> dict | None:
    """État à reprendre, ou None (aucun, trop vieux, ou trop de tentatives : il est alors abandonné)."""
    entry = _load(channel).get(channel)
    if not entry or not entry.get("topic"):
        return None
    try:
        saved = datetime.fromisoformat(entry["saved_at"])
    except (KeyError, TypeError, ValueError):
        saved = datetime.now(UTC)
    if int(entry.get("script_attempts", 0)) >= max_attempts or \
            datetime.now(UTC) - saved > timedelta(days=max_age_days):
        print(f"🗑️ Sujet en attente abandonné ({entry['topic']}) : trop de tentatives ou trop ancien.")
        clear_pending(channel)
        return None
    return entry


def clear_pending(channel: str) -> None:
    data = _load(channel)
    if data.pop(channel, None) is not None:
        _write(data, channel)


# ---------- Sujets abandonnés (contrôle de véracité ou matière insuffisante) ----------
REJECTED_KEY = "_rejected"     # aucune chaîne ne commence par « _ » : pas de collision avec les entrées
MAX_REJECTED = 300


def note_rejected(channel: str, source_title: str | None, anecdote_id: str | None = None,
                  reason: str = "") -> None:
    """Le sujet ne sera plus proposé : ni sa page source, ni son anecdote."""
    if not source_title and not anecdote_id:
        return
    data = _load(channel)
    entries = list((data.get(REJECTED_KEY) or {}).get(channel) or [])
    entries.append({"source_title": source_title, "anecdote_id": anecdote_id, "reason": reason[:300],
                    "date": datetime.now(UTC).isoformat(timespec="seconds")})
    data.setdefault(REJECTED_KEY, {})[channel] = entries[-MAX_REJECTED:]
    _write(data, channel)


def load_rejected(channel: str) -> tuple[set[str], set[str]]:
    """(titres des pages en minuscules, identifiants d'anecdotes) des sujets abandonnés."""
    entries = (_load(channel).get(REJECTED_KEY) or {}).get(channel) or []
    return ({str(e["source_title"]).lower() for e in entries if e.get("source_title")},
            {str(e["anecdote_id"]) for e in entries if e.get("anecdote_id")})
