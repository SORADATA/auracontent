"""mystery, run 2 : reveal_sentence choisie par le code, comptage des scènes (plage + cible réduite), JSON tronqué,
abandon après 2 échecs, titres normalisés (Poissons, Boussac, Stenay), rapport de fin."""
import json

import pytest
from auracontent.channels.base_channel import load_channel_config
from auracontent.channels.mystery.brain import MysteryBrain, TitleSet, similarity, title_key
from auracontent.core.brain.base_brain import SceneCountError
from auracontent.core.brain.llm_client import LLMClient, _json_truncated, _repair_json_text
from auracontent.services.visuals.subject_images import mentions_focus


def _mystery(llm=None) -> MysteryBrain:
    return MysteryBrain(config=load_channel_config("mystery"), llm=llm)


# ---------- 1. reveal_sentence : choisie par le code ----------
SCENES = [{"role": "hook", "text": "Une statue cache un secret."},
          {"role": "context", "text": "Elle date de 1865. Elle est dans l'église."},
          {"role": "reveal", "text": "Regarde bien. Le saint a le visage de Vercingétorix."},
          {"role": "cta", "text": "Et toi, tu l'avais vu ?"}]


def _data():
    return {"reveal_sentence": "inventée", "scenes": [dict(s) for s in SCENES]}


def test_reveal_is_the_sentence_closest_to_the_answer():
    brain = _mystery()
    brain._source = {"answer": "Le visage du saint est celui de Vercingétorix."}
    data = _data()
    brain.locate_reveal(data)
    assert data["scenes"][2]["reveal_sentence"] == "Le saint a le visage de Vercingétorix."
    assert data["reveal_sentence"] == "Le saint a le visage de Vercingétorix."
    assert sum("reveal_sentence" in s for s in data["scenes"]) == 1


def test_reveal_falls_back_to_last_sentence_of_reveal_scene():
    brain = _mystery()
    brain._source = {"answer": "Un notaire l'a achetée aux enchères."}      # aucune phrase proche
    data = _data()
    data["scenes"][2]["text"] = "Regarde bien. Il y a un détail."
    brain.locate_reveal(data)
    assert data["scenes"][2]["reveal_sentence"] == "Il y a un détail."


@pytest.mark.parametrize("source", [None, {}, {"answer": ""}])
def test_reveal_never_fails(source):
    brain = _mystery()
    brain._source = source
    data = {"scenes": [{"role": "context", "text": "Un."}, {"role": "context", "text": "Deux. Trois."},
                       {"role": "cta", "text": "Quatre ?"}]}
    brain.locate_reveal(data)                                            # pas de scène « reveal »
    assert data["scenes"][1]["reveal_sentence"] == "Trois."
    brain.locate_reveal({"scenes": []})                                  # rien à choisir : aucune erreur


def test_script_prompt_no_longer_asks_reveal_sentence():
    brain = _mystery()
    brain._source = {"title": "X", "extract": "y"}
    messages = brain.build_script_prompt("t", 5, {"text": "h"})
    assert "reveal_sentence" not in messages[0]["content"] + messages[1]["content"]


def test_similarity_ignores_case_and_accents():
    assert similarity("Le SAINT a le visage de Vercingetorix", "le saint a le visage de Vercingétorix") == 1.0
    assert similarity("", "x") == 0.0


# ---------- 2. Comptage des scènes ----------
def test_scene_count_range_and_reduced_targets():
    brain = _mystery()
    assert [n for n in range(1, 10) if brain.scene_count_accepted(n, 6)] == [4, 5, 6, 7]
    assert brain.scene_count_candidates(6) == [6, 5, 4]
    assert brain.scene_count_candidates(4) == [4]


class _CountLLM:
    def __init__(self, *counts):
        self.counts, self.targets = list(counts), []

    def call_json_with_retry(self, messages, **kw):
        n = self.counts.pop(0)
        return {"scenes": [{"id": 10 + i, "text": f"Phrase {i}."} for i in range(n)]}


def test_out_of_range_retries_with_reduced_target_then_renumbers(monkeypatch, capsys):
    monkeypatch.delenv("VIDEO_DURATION", raising=False)
    brain = _mystery(_CountLLM(9, 8, 5))                # 9 pour 6, 8 pour 5 : hors plage ; 5 pour 4 : accepté
    targets = []
    brain.build_script_prompt = lambda topic, n, hook, feedback="": targets.append(n) or [{"role": "user",
                                                                                          "content": "x"}]
    brain._validate_script_base = lambda data: None
    brain.validate_script_specifics = lambda data, topic: None
    brain.check_script_logic = lambda topic, data: []
    brain.target_scene_count = lambda: 6
    data = brain.generate_script("t")
    assert targets == [6, 5, 4]                         # aucune correction au même nombre : cible réduite
    assert [s["id"] for s in data["scenes"]] == [1, 2, 3, 4, 5]
    assert "accepté et renuméroté" in capsys.readouterr().out


def test_all_targets_out_of_range_is_a_script_failure(monkeypatch):
    brain = _mystery(_CountLLM(9, 9, 9, 9, 9))
    targets = []
    brain.build_script_prompt = lambda topic, n, hook, feedback="": targets.append((n, bool(feedback))) or [
        {"role": "user", "content": "x"}]
    brain.target_scene_count = lambda: 6
    with pytest.raises(RuntimeError) as err:
        brain.generate_script("t")
    assert isinstance(err.value.__cause__, SceneCountError)
    # 6 et 5 : un essai chacun puis cible réduite ; 4 (dernière cible) : 3 essais, avec correction demandée
    assert targets == [(6, False), (5, False), (4, False), (4, True), (4, True)]


def test_last_target_gets_corrections_not_a_single_try():
    """Opéra de Lyon : matière courte, cible 4 (déjà le minimum) ; le LLM rend 7 scènes puis corrige."""
    brain = _mystery(_CountLLM(7, 4))
    brain.build_script_prompt = lambda *a, **k: [{"role": "user", "content": "x"}]
    brain._validate_script_base = lambda data: None
    brain.validate_script_specifics = lambda data, topic: None
    brain.check_script_logic = lambda topic, data: []
    brain.target_scene_count = lambda: 4
    assert len(brain.generate_script("t")["scenes"]) == 4


# ---------- 4. Mots d'un titre : ponctuation et milliers ----------
def test_title_word_count_ignores_punctuation_and_thousands():
    from auracontent.channels.mystery.brain import title_words

    ecrins = "1864 : La Barre des Écrins, 4\u202f101\u202fm, était\u2011elle déjà hors du top français ?"
    assert title_words(ecrins) == 13                      # le run comptait 16 (« : », « ? », « 4 101 »)
    assert MysteryBrain.title_format_issue(ecrins) is None
    assert title_words(" ".join(["mot"] * 15)) == 15


def test_other_channels_keep_correction_at_same_target():
    """finance : comptage exact, correction demandée au même nombre (aucune cible réduite)."""
    from auracontent.channels.base_channel import BaseChannel
    from auracontent.channels.finance.brain import FinanceBrain

    finance = BaseChannel("finance", FinanceBrain, None).brain
    assert finance.reduce_target_on_count_mismatch is False
    assert finance.scene_count_accepted(7, 7) and not finance.scene_count_accepted(6, 7)
    assert finance.scene_count_candidates(7) == [7]


# ---------- 2. JSON tronqué : jamais refermé en silence ----------
FULL = {"title": "T", "scenes": [{"id": i, "text": f"Phrase {i}."} for i in range(1, 7)], "closing_cta": "?"}
CUT = json.dumps(FULL)[:120].rsplit("},", 1)[0] + "},"     # coupé juste après une scène complète


def test_truncated_json_is_detected():
    assert _json_truncated(CUT)
    repaired = json.loads(_repair_json_text(CUT))
    assert len(repaired["scenes"]) < 6 and "closing_cta" not in repaired    # ce qu'on perdait en silence
    assert not _json_truncated(json.dumps(FULL)) and not _json_truncated('{"a": 1,}')


def test_truncated_json_is_asked_again_with_more_tokens():
    client = LLMClient()
    answers, budgets = [CUT, json.dumps(FULL)], []

    def fake(messages, **kw):
        budgets.append(kw["max_completion_tokens"])
        return answers.pop(0)

    client.call_with_fallback = fake
    data = client.call_json_with_retry([{"role": "user", "content": "x"}], max_completion_tokens=3000)
    assert len(data["scenes"]) == 6 and data["closing_cta"] == "?"
    assert budgets == [3000, 4000]


def test_truncated_json_on_last_attempt_is_still_repaired():
    client = LLMClient()
    client.call_with_fallback = lambda messages, **kw: CUT
    data = client.call_json_with_retry([{"role": "user", "content": "x"}])
    assert 0 < len(data["scenes"]) < 6                  # dernier recours : le comptage décide ensuite


# ---------- 3. Sujet en attente : abandonné après 2 échecs du script ----------
def _global(monkeypatch, tmp_path, error):
    from auracontent.core.brain import global_brain as gb
    from auracontent.utils import pending_topic as pt

    monkeypatch.setattr(pt, "PENDING_PATH", tmp_path / "pending.json")

    class Brain:
        last_pick: dict | None = None

        def pick_topic(self):
            self.last_pick = {"source_title": "Opéra de Lyon", "anecdote_id": "42"}
            return "Opéra de Lyon"

        def generate_hook_variants(self, topic, **k):
            return [{"text": "accroche", "pattern": "chiffre"}]

        def generate_script(self, topic, chosen_hook=None):
            raise error

        def export_state(self):
            return {"last_pick": self.last_pick}

        def restore_state(self, state):
            self.last_pick = dict(state.get("last_pick") or {})

    class Channel:
        brain = Brain()

        def section(self, key):
            return {"brain": {"veracity_topic_rounds": 3, "resume_max_attempts": 2}}.get(key, {})

    g = gb.GlobalBrain.__new__(gb.GlobalBrain)
    g.channels, g.output_dir = {"mystery": Channel()}, tmp_path
    monkeypatch.setattr(gb, "compute_pattern_scores", lambda *a: {})
    monkeypatch.setattr(gb, "load_hook_history", lambda c: [])
    return g, pt


@pytest.mark.parametrize("error", [ValueError("'closing_cta' manquant"), RuntimeError("comptage des scènes")])
def test_any_script_validation_failure_counts_and_abandons_after_two(monkeypatch, tmp_path, capsys, error):
    g, pt = _global(monkeypatch, tmp_path, error)
    with pytest.raises(type(error)):
        g.run_channel("mystery", previous_stats_list=[])          # 1er échec : sujet gardé
    assert pt.load_pending("mystery", max_attempts=2)["script_attempts"] == 1
    assert "(1/2 échec(s))" in capsys.readouterr().out
    with pytest.raises(type(error)):
        g.run_channel("mystery", previous_stats_list=[])          # reprise, 2e échec : abandonné tout de suite
    assert "abandonné après 2 échec(s)" in capsys.readouterr().out
    assert pt.load_pending("mystery", max_attempts=2) is None
    titles, ids = pt.load_rejected("mystery")
    assert titles == {"opéra de lyon"} and ids == {"42"}          # plus jamais proposé


def test_mystery_yaml_abandons_after_two_failures():
    assert load_channel_config("mystery")["brain"]["resume_max_attempts"] == 2


# ---------- 4. Titres normalisés ----------
@pytest.mark.parametrize("raw, key", [
    ("Poissons (Haute-Marne)", "poissons"),
    ("Boussac (Creuse)", "boussac"),
    ("Château\u00a0de\u202fBoussac", "chateau de boussac"),
    ("Saint\u2011Dizier", "saint dizier"),
    ("ÉGLISE Notre-Dame", "eglise notre dame"),
    ("\ufb01lature (Ariège)", "filature"),                      # NFKC : ligature « \ufb01 »
])
def test_title_key(raw, key):
    assert title_key(raw) == key


def test_title_set_ignores_parentheses_accents_case():
    used = TitleSet({"poissons (haute-marne)", "Boussac (Creuse)"})
    assert "Poissons" in used and "POISSONS (Haute-Marne)" in used and "Boussac" in used
    assert "Boussac-Bourg" not in used
    assert "poissons" in (used | {"Stenay"}) and "Stenay" in (used | {"Stenay"})


@pytest.mark.parametrize("article, title", [
    ("Poissons (Haute-Marne)", "Pourquoi Poissons a-t-il gardé sa halle ?"),
    ("Boussac (Creuse)", "Comment Boussac a-t-il caché la Dame à la licorne ?"),
])
def test_title_naming_the_article_passes(article, title):
    commune = article.split(" (")[0]
    trial = {"source": {"title": article}, "focus": [commune],
             "details": {"commune": commune, "department": article.split("(")[1].rstrip(")")}}
    assert not mentions_focus(title, trial["focus"], [commune])     # l'ancien contrôle rejetait ce titre
    assert MysteryBrain.names_subject(title, trial)


def test_title_naming_short_object_name_passes():
    trial = {"source": {"title": "Armorial des communes de la Meuse"}, "focus": ["blason de la ville de Stenay"],
             "details": {"commune": "Stenay", "department": "Meuse"}}
    assert MysteryBrain.names_subject("Pourquoi Stenay a-t-elle un diable sur son blason ?", trial)
    assert MysteryBrain.names_subject("Ce blason cache un diable", trial)          # nom commun, comme avant
    assert not MysteryBrain.names_subject("Une ville de la Meuse cache un diable", trial)


def test_anecdote_already_used_is_skipped_despite_parentheses(monkeypatch):
    from auracontent.channels.mystery import brain as mb

    brain = _mystery()
    brain._anecdotes = [{"target": "Poissons (Haute-Marne)", "id": "1", "text": "a"},
                        {"target": "Stenay", "id": "2", "text": "b"}]
    brain._rejected_ids = set()
    fetched = []
    monkeypatch.setattr(mb, "fetch_page", lambda title: fetched.append(title) or {"title": title, "url": "u"})
    page = brain._next_anecdote({"poissons"})
    assert page["title"] == "Stenay" and fetched == ["Stenay"]


# ---------- 5. Rapport de fin ----------
def test_run_report_counts_tested_and_rejections_by_reason():
    brain = _mystery()
    assert brain.run_report() == []
    brain._report.update({"sujets testés": 10, "rejet : retournement": 4, "rejet : images": 3,
                          "rejet : titre": 2, "rejet : faits": 1, "rejet : script": 1, "repli": 1})
    line = brain.run_report()[0]
    assert "10 testé(s)" in line and "1 passage(s) en repli" in line
    assert "retournement 4, images 3, titre 2, faits 1, script 1" in line


def test_llm_calls_are_reported_by_stage():
    client = LLMClient()
    with client.in_stage("retournement"):
        client._record("Groq", True)
        client._record("Groq", False)
    with client.in_stage("titre"):
        client._record("Groq", True)
    client._record("Gemini", True)
    report = "\n".join(client.usage_report())
    assert "Appels LLM par étape : retournement 2 (dont 1 échoué(s)), titre 1, autre 1" in report
    assert client.stage == "autre"
