"""mystery : sujet abandonné sur le comptage des scènes, matière courte (4 scènes au moins), repli sans contrôle
du retournement, bascule au premier 429, motif de chaque titre rejeté."""
import pytest
from auracontent.channels.base_channel import load_channel_config
from auracontent.channels.mystery.brain import MysteryBrain
from auracontent.core.brain import base_brain
from auracontent.core.brain.base_brain import BaseBrain, SceneCountError, TopicRejected


def _mystery(llm=None) -> MysteryBrain:
    return MysteryBrain(config=load_channel_config("mystery"), llm=llm)


# ---------- 1. Comptage des scènes : sujet abandonné tout de suite ----------
class _WrongCountLLM:
    """Renvoie toujours 9 scènes, quelle que soit la cible."""

    def __init__(self):
        self.calls = 0

    def call_json_with_retry(self, messages, **kw):
        self.calls += 1
        return {"scenes": [{"id": i, "text": "Une phrase."} for i in range(1, 10)]}


def test_scene_count_failure_raises_scene_count_error(monkeypatch):
    monkeypatch.delenv("VIDEO_DURATION", raising=False)
    brain = _mystery(_WrongCountLLM())
    brain.build_script_prompt = lambda *a, **k: [{"role": "user", "content": "x"}]
    with pytest.raises(RuntimeError) as err:
        BaseBrain.generate_script_with_target(brain, "t", scene_count=5, max_retries=1)
    assert isinstance(err.value.__cause__, SceneCountError)


def test_scene_count_failure_is_a_normal_script_failure(monkeypatch):
    """Plus d'abandon immédiat sur le comptage (cible réduite d'abord) : l'échec compte pour l'abandon du sujet
    en attente, comme tout autre échec de validation."""
    brain = _mystery()

    def fail(self, topic, **k):
        raise RuntimeError("Impossible de générer un script.") from SceneCountError("comptage")

    monkeypatch.setattr(BaseBrain, "generate_script_with_target", fail)
    monkeypatch.setattr(MysteryBrain, "target_scene_count", lambda self: 5)
    with pytest.raises(RuntimeError) as err:
        brain.generate_script("Opéra de Lyon")
    assert not isinstance(err.value, TopicRejected)
    assert brain._report["rejet : script"] == 1


@pytest.mark.parametrize("error", [RuntimeError("quota"), TopicRejected("fait absent de la source")])
def test_other_script_failures_unchanged(monkeypatch, error):
    brain = _mystery()

    def fail(self, topic, **k):
        raise error

    monkeypatch.setattr(BaseBrain, "generate_script_with_target", fail)
    monkeypatch.setattr(MysteryBrain, "target_scene_count", lambda self: 5)
    with pytest.raises(type(error)) as err:
        brain.generate_script("t")
    assert err.value is error                                       # le sujet reste repris s'il n'est pas rejeté


def test_scene_count_rejection_drops_pending_and_moves_on(monkeypatch, tmp_path):
    """Opéra de Lyon en attente : repris une fois, rejeté, plus jamais proposé ; le sujet suivant est traité."""
    from auracontent.core.brain import global_brain as gb
    from auracontent.utils import pending_topic as pt

    monkeypatch.setattr(pt, "PENDING_PATH", tmp_path / "pending.json")
    pt.save_pending("mystery", {"topic": "Opéra de Lyon", "hooks": [], "chosen_hook": {"text": "h"},
                                "brain": {"last_pick": {"source_title": "Opéra de Lyon", "anecdote_id": "42"}}})

    class Brain:
        last_pick: dict | None = None

        def pick_topic(self):
            self.last_pick = {"source_title": "Abbaye", "anecdote_id": "7"}
            return "Abbaye"

        def generate_hook_variants(self, topic, **k):
            return [{"text": "accroche", "pattern": "chiffre"}]

        def generate_script(self, topic, chosen_hook=None):
            if topic == "Opéra de Lyon":
                raise TopicRejected("nombre de scènes non tenu")
            return {"scenes": []}

        def export_state(self):
            return {}

        def restore_state(self, state):
            self.last_pick = dict(state.get("last_pick") or {})

    class Channel:
        brain = Brain()

        def section(self, key):
            return {"brain": {"veracity_topic_rounds": 3}}.get(key, {})

    g = gb.GlobalBrain.__new__(gb.GlobalBrain)
    g.channels, g.output_dir = {"mystery": Channel()}, tmp_path
    monkeypatch.setattr(gb, "save_global_history", lambda *a, **k: None)
    monkeypatch.setattr(gb, "compute_pattern_scores", lambda *a: {})
    monkeypatch.setattr(gb, "load_hook_history", lambda c: [])
    path = g.run_channel("mystery", previous_stats_list=[])
    assert '"topic": "Abbaye"' in path.read_text(encoding="utf-8")
    titles, ids = pt.load_rejected("mystery")
    assert titles == {"opéra de lyon"} and ids == {"42"}
    assert pt.load_pending("mystery") is None


def test_base_brain_exports_scene_count_error():
    assert issubclass(base_brain.SceneCountError, ValueError)        # même traitement qu'avant pour les autres


# ---------- 2. Matière courte : 4 scènes au moins ----------
def test_short_material_never_below_four_scenes(monkeypatch):
    monkeypatch.delenv("VIDEO_DURATION", raising=False)
    brain = _mystery()
    assert brain.min_scenes() == 4
    brain._source = {"facts": ["f"] * 4}
    assert brain.target_scene_count() == 4
    brain._source = {"facts": ["f"] * 9}
    assert brain.target_scene_count() > 4


class _Facts:
    def __init__(self, n):
        self.n = n

    def call_json_with_retry(self, messages, **kw):
        return {"faits": [f"fait {i}" for i in range(self.n)], "reponse": "Parce que."}


def test_too_few_facts_for_four_scenes_rejects_before_script(capsys):
    trial = {"source": {"title": "Opéra de Lyon", "extract": "x"}, "focus": ["Opéra"]}
    brain = _mystery(_Facts(3))
    brain.config["brain"]["min_facts"] = 2                         # même avec un min_facts plus bas
    assert brain.extract_facts(dict(trial), "t") is False
    assert "4 scènes au moins" in capsys.readouterr().out
    assert _mystery(_Facts(4)).extract_facts(dict(trial), "t") is True


def test_tolerance_never_goes_below_floor():
    brain = _mystery()
    data = {"scenes": [{"id": i, "text": "x"} for i in range(1, 4)], "closing_cta": "?"}
    with pytest.raises(SceneCountError, match="au moins 4"):
        brain.validate_script_specifics(data, "t")


# ---------- 4. Bascule au premier 429 ----------
def test_mystery_rate_limit_fallback_on_every_route():
    brain = _mystery()
    assert brain.config["brain"]["rate_limit"] == "fallback"
    assert brain.script_routing()["rate_limit"] == "fallback"
    assert brain.simple_routing()["rate_limit"] == "fallback"
    assert brain.script_routing()["wait_rounds"] == 3              # saturation générale : on attend toujours


# ---------- 5. Motif de chaque titre rejeté ----------
@pytest.mark.parametrize("title, reason", [
    ("", "titre vide"),
    ("x" * 90, "90 caractères"),
    (" ".join(["mot"] * 15), "15 mots"),
    ("Le plus vieux pont de France", "mot interdit « le plus »"),
    ("Une histoire incroyable", "mot interdit « incroyable »"),
])
def test_title_format_rejection_always_has_reason(title, reason):
    assert reason in MysteryBrain.title_format_issue(title)


def test_valid_title_has_no_format_issue():
    assert MysteryBrain.title_format_issue("Pourquoi cette abbaye est-elle devenue une prison ?") is None


class _TitleLLM:
    def __init__(self, title):
        self.title = title

    def call_json_with_retry(self, messages, **kw):
        return {"title": self.title}


def test_write_title_logs_reason(capsys):
    brain = _mystery(_TitleLLM("Le plus vieux pont de France"))
    title_vars = {k: "" for k in ("forced_region", "forced_category", "forced_format")}
    assert brain._write_title(title_vars, {"title": "Pont", "extract": "x"}) is None
    out = capsys.readouterr().out
    assert "Titre rejeté (mot interdit « le plus »)" in out


def test_every_title_rejection_line_carries_a_reason():
    """Toutes les lignes « Titre rejeté » du brain précisent un motif (pas de rejet muet)."""
    import inspect
    import re

    source = inspect.getsource(MysteryBrain)
    lines = [ln for ln in source.splitlines() if "Titre" in ln and "rejeté" in ln and "print(" in ln]
    assert len(lines) >= 4
    assert all(re.search(r"\(\{|ne nomme|\(\{reason", ln) for ln in lines), lines


# ---------- 3. Repli : le sujet le mieux illustré, sans repasser le contrôle du retournement ----------
def _trial(title="Abbaye de Clairvaux"):
    return {"source": {"title": title, "url": "u", "extract": f"{title} devint une prison en 1808."},
            "focus": [], "details": {}, "images": 9, "focus_images": 9, "region": {"name": "Grand Est"},
            "category": {"name": "abbaye"}, "title_vars": {}}


def test_accept_title_without_reversal_check_never_calls_judge():
    brain = _mystery()
    brain._write_title = lambda *a, **k: "Abbaye de Clairvaux devint une prison en 1808"
    brain.is_duplicate_topic_llm = lambda title, used: (False, None)
    brain.extract_facts = lambda trial, title: True

    def judge(*a):
        raise AssertionError("le repli ne repasse pas le contrôle du retournement")

    brain.has_reversal = judge
    assert brain._accept_title(_trial(), [], check_reversal=False) == "Abbaye de Clairvaux devint une prison en 1808"
    brain.has_reversal = lambda title, source: (False, "simple fait")
    assert brain._accept_title(_trial(), []) is None                 # boucle normale : contrôle appliqué


def test_fallback_skips_reversal_but_keeps_other_checks(monkeypatch, capsys):
    from auracontent.channels.mystery import brain as mb

    monkeypatch.setattr(mb, "load_global_history", lambda: [])
    monkeypatch.setattr(mb, "load_used_sources", lambda: set())
    monkeypatch.setattr(mb, "load_last_regions", lambda **k: [])
    monkeypatch.setattr(mb, "load_last_categories", lambda **k: [])
    monkeypatch.setattr(mb, "place_details", lambda qid: {"commune": "C"})
    pages = iter([{"title": f"Lieu {i}", "url": "u", "extract": "x"} for i in range(10)])
    brain = _mystery()
    brain._anecdotes = []
    brain._find_source = lambda region, category, used: next(pages)
    brain.count_source_images = lambda source, details, focus=None: (6 + int(source["title"][-1]),) * 2
    calls = []

    def accept(trial, used, check_reversal=True):
        calls.append((trial["source"]["title"], check_reversal))
        return None if check_reversal else f"Titre {trial['source']['title']}"   # le juge rejette tout

    brain._accept_title = accept
    assert brain.pick_topic() == "Titre Lieu 9"                   # le mieux illustré (15 images)
    assert calls[:10] == [(f"Lieu {i}", True) for i in range(10)] and calls[10] == ("Lieu 9", False)
    assert "sans repasser le contrôle du retournement" in capsys.readouterr().out
