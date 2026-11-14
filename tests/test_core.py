"""Tests rapides (< 30 s), sans clé API : lancés en CI à chaque push."""
import json
import math
import random
import shutil

import pytest

from auracontent.core.brain.llm_client import _classify_error, _repair_json_text
from auracontent.services.audios.text_fr import normalize_fr
from auracontent.utils import hook_tracker
from auracontent.utils.subtitles import generate_grouped_srt


# ---------- Sous-titres ----------
def test_srt_covers_whole_duration(tmp_path):
    path = generate_grouped_srt("un deux trois quatre cinq six sept", 3.5, str(tmp_path / "a.srt"))
    blocks = open(path, encoding="utf-8").read().strip().split("\n\n")
    assert len(blocks) == 3  # groupes de 3 mots
    assert blocks[0].splitlines()[1].startswith("00:00:00,000")
    assert blocks[-1].splitlines()[1].endswith("00:00:03,500")


def test_srt_empty_text_returns_none(tmp_path):
    assert generate_grouped_srt("", 3, str(tmp_path / "a.srt")) is None


# ---------- Texte pour la voix ----------
@pytest.mark.parametrize("raw, expected", [
    ("Au XIVe siècle", "Au quatorzième siècle"),
    ("Louis XIV régna", "Louis quatorze régna"),
    ("Napoléon Ier", "Napoléon premier"),
    ("300 av. J.-C.", "300 avant Jésus-Christ"),
    ("à 12 km", "à 12 kilomètres"),
    ("« Le vol »", '"Le vol"'),
])
def test_normalize_fr(raw, expected):
    assert normalize_fr(raw) == expected


# ---------- LLM ----------
@pytest.mark.parametrize("message, kind", [
    ("Error code: 400 - Failed to validate JSON. See 'failed_generation'", "json_failed"),
    ("Error code: 400 - The model `x` has been decommissioned", "not_found"),
    ("Error code: 404 - This model is unavailable for free", "not_found"),
    ("Error code: 402 - This request requires more credits", "no_credit"),
    ("Error code: 429 - Rate limit reached", "rate_limit"),
])
def test_classify_llm_errors(message, kind):
    assert _classify_error(Exception(message), message) == kind


def test_repair_truncated_json():
    broken = '{"title":"x","scenes":[{"id":1,"text":"a"},"id":2,"text":"b'
    data = json.loads(_repair_json_text(broken))
    assert [s["id"] for s in data["scenes"]] == [1, 2]


def test_repair_strips_reasoning():
    assert json.loads(_repair_json_text('<think>hmm</think>{"ok": true}')) == {"ok": True}


# ---------- Bandit des hooks ----------
def test_bandit_exploits_best_pattern(tmp_path, monkeypatch):
    monkeypatch.setattr(hook_tracker, "HOOK_HISTORY_PATH", tmp_path / "hooks.json")
    hook_tracker.record_hook_usage("Trésor perdu", "question", "mystery", "🔍 Trésor perdu... Qui l'a caché ?")
    hook_tracker.record_hook_usage("Cloche", "chiffre", "mystery", "🔔 En 1792, une cloche sonna seule")
    stats = [{"title": "🔍 Trésor perdu... Qui l'a caché ? #mystère", "views": 9000, "likes": 700},
             {"title": "🔔 En 1792, une cloche sonna seule", "views": 100, "likes": 2}]
    scores = hook_tracker.compute_pattern_scores(stats, hook_tracker.load_hook_history("mystery"))
    assert scores["question"] > scores["chiffre"]
    hooks = [{"text": "a", "pattern": "chiffre"}, {"text": "b", "pattern": "question"}]
    assert hook_tracker.select_hook(hooks, scores, epsilon=0.0)["pattern"] == "question"


def test_bandit_without_scores_still_picks():
    random.seed(0)
    assert hook_tracker.select_hook([{"text": "a", "pattern": "x"}], {}, epsilon=0.0)["text"] == "a"


# ---------- Montage (ffmpeg) ----------
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_ken_burns_scene_has_expected_duration(tmp_path):
    """Régression : une scène image de 5 s sortait en 417 s (zoompan sur une image répétée à 25 fps)."""
    import ffmpeg
    from PIL import Image

    from auracontent.services.video.composer import Composer

    image = tmp_path / "img.jpg"
    Image.new("RGB", (1200, 800), (90, 60, 30)).save(image)
    voice = tmp_path / "voice.wav"
    ffmpeg.input("sine=frequency=440:duration=2", f="lavfi").output(str(voice), ac=2, ar=48000).run(quiet=True)

    composer = Composer({"video_width": 360, "video_height": 640, "fps": 15}, channel_name="mystery")
    composer.temp_dir = str(tmp_path)
    out = composer.process_scene({"id": "t", "audio_path": str(voice), "duration": 2.0}, str(image))

    assert out is not None
    duration = float(ffmpeg.probe(out)["format"]["duration"])
    assert 1.9 <= duration <= 2.6  # 2 s + SECOND_IMAGE_EXTRA_DURATION (0.35 s)


# ---------- Qualité / audience ----------
@pytest.mark.parametrize("duration, expected", [("45", 6), ("30", 5), ("90", 11), ("200", 11), ("abc", 6)])
def test_scene_count_follows_target_duration(monkeypatch, duration, expected):
    from auracontent.core.brain.base_brain import BaseBrain

    brain = BaseBrain.__new__(BaseBrain)
    brain.config = {"brain": {"scene_count": 11, "seconds_per_scene": 8}}
    monkeypatch.setenv("VIDEO_DURATION", duration)
    assert brain.target_scene_count() == expected


def test_best_scored_hook_wins_without_stats():
    hooks = [{"text": "a", "pattern": "x", "score": 6}, {"text": "b", "pattern": "y", "score": 9}]
    assert hook_tracker.select_hook(hooks, {}, epsilon=0.0)["text"] == "b"


@pytest.mark.parametrize("body, expected", [
    ('{"error": {"details": [{"retryDelay": "37s"}]}}', 38.0),
    ('{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}', None),
    ('{"error": "quota"}', 30.0),
])
def test_gemini_tts_retry_delay(body, expected):
    from auracontent.services.audios.audio import AudioEngine

    assert AudioEngine._retry_delay(body) == expected


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_landscape_image_is_not_distorted(tmp_path):
    """Régression : zoompan étirait les photos horizontales au format 9:16 (un cercle devenait une ellipse)."""
    import ffmpeg
    from PIL import Image, ImageDraw

    from auracontent.services.video.composer import Composer

    image = tmp_path / "circle.jpg"
    img = Image.new("RGB", (1600, 1000), (0, 0, 0))
    ImageDraw.Draw(img).ellipse((500, 200, 1100, 800), fill="white")
    img.save(image)
    voice = tmp_path / "voice.wav"
    ffmpeg.input("sine=duration=1", f="lavfi").output(str(voice), ac=2, ar=48000).run(quiet=True)

    composer = Composer({"video_width": 360, "video_height": 640, "fps": 10}, channel_name="mystery")
    composer.temp_dir = str(tmp_path)
    out = composer.process_scene({"id": "c", "audio_path": str(voice), "duration": 1.0}, str(image))
    frame = tmp_path / "frame.png"
    ffmpeg.input(out, ss=0.05).output(str(frame), vframes=1).run(quiet=True)

    px = Image.open(frame).convert("L")
    w, h = px.size
    bright = [(x, y) for x in range(w) for y in range(int(h * 0.3), int(h * 0.7)) if px.getpixel((x, y)) > 200]
    xs, ys = [p[0] for p in bright], [p[1] for p in bright]
    ratio = (max(xs) - min(xs)) / (max(ys) - min(ys))
    assert 0.85 <= ratio <= 1.15  # toujours un cercle


# ---------- Limites de débit ----------
class _FakeError(Exception):
    def __init__(self, msg, body=None):
        super().__init__(msg)
        self.body = body


def test_retry_after_groq_message():
    from auracontent.core.brain.llm_client import _retry_after

    msg = "Error code: 429 - Rate limit reached on tokens per minute (TPM): Limit 8000, Used 6844, Requested 2100. Please try again in 7.35s."
    assert _retry_after(_FakeError(msg), msg) == pytest.approx(7.35)


def test_retry_after_gemini_body():
    from auracontent.core.brain.llm_client import _is_daily_quota, _retry_after

    body = [{"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "23s"}]}}]
    e = _FakeError("Error code: 429", body)
    assert _retry_after(e, str(e)) == 23
    assert not _is_daily_quota(e, str(e))


def test_gemini_daily_quota_detected():
    from auracontent.core.brain.llm_client import _is_daily_quota

    body = [{"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}]
    e = _FakeError("Error code: 429", body)
    assert _is_daily_quota(e, str(e))


def test_groq_tpm_message_with_used_field():
    from auracontent.core.brain.llm_client import _RATE_LIMIT_NUMS_RE

    m = _RATE_LIMIT_NUMS_RE.search("Limit 8000, Used 4521, Requested 9100")
    assert (m.group(1), m.group(2)) == ("8000", "9100")


def test_duplicate_check_skips_llm_when_no_overlap():
    from auracontent.core.brain.base_brain import BaseBrain

    class NoCallLLM:
        def call_json_with_retry(self, *a, **k):
            raise AssertionError("le LLM ne doit pas être appelé")

    brain = BaseBrain.__new__(BaseBrain)
    brain.llm = NoCallLLM()
    history = ["Le trésor perdu de Rennes-le-Château", "La cloche fantôme de Bretagne"]
    assert brain.is_duplicate_topic_llm("Un médecin empoisonneur à Besançon", history) == (False, None)


def test_llm_registry_skips_providers_without_key(monkeypatch):
    from auracontent.core.brain import llm_client

    for spec in llm_client.PROVIDER_SPECS.values():
        for var in spec["key_env"]:
            monkeypatch.delenv(var, raising=False)
    for var in ("LLM_PROVIDERS",):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")  # sans CLOUDFLARE_ACCOUNT_ID -> ignoré
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    client = llm_client.LLMClient.create_default()
    assert {p["name"] for p in client.providers} == {"DeepSeek"}


# ---------- Narration Gemini en une seule requête ----------
AUDIO_CFG = {
    "engines": ["gemini", "edge"],
    "gemini": {"narrator": {"voice": "Charon", "style": "calm"}, "witness": {"voice": "Orus", "style": "tense"}},
    "edge": {"narrator": {"voice": "fr-FR-HenriNeural"}, "witness": {"voice": "fr-FR-HenriNeural"}},
}


def test_split_points_snap_to_paragraph_pauses():
    from auracontent.services.audios.audio import AudioEngine

    # 3 paragraphes (poids 1, 2, 1) sur 20 s ; pauses longues à 5.0 et 15.0, pause courte parasite à 9.0
    silences = [(4.5, 5.5), (8.9, 9.1), (14.6, 15.4)]
    assert AudioEngine.split_points(20.0, [1, 2, 1], silences) == [5.0, 15.0]


def test_split_points_gives_up_without_pauses():
    from auracontent.services.audios.audio import AudioEngine

    assert AudioEngine.split_points(20.0, [1, 1, 1], [(1.0, 1.4)]) is None


def test_batch_prompt_uses_two_voices_only_when_needed(monkeypatch, tmp_path):
    from auracontent.services.audios.audio import AudioEngine

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    engine = AudioEngine(AUDIO_CFG, output_dir=str(tmp_path))
    solo = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    _, speech = engine._batch_prompt(solo, ["a", "b"])
    assert speech["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Charon"

    duo = [{"id": 1, "text": "a"}, {"id": 2, "text": "b", "voice_type": "witness"}]
    prompt, speech = engine._batch_prompt(duo, ["a", "b"])
    voices = {c["speaker"]: c["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]
              for c in speech["multiSpeakerVoiceConfig"]["speakerVoiceConfigs"]}
    assert voices == {"Narrateur": "Charon", "Temoin": "Orus"}
    assert "Temoin: b" in prompt


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_single_request_narration_is_split_per_scene(monkeypatch, tmp_path):
    import asyncio

    import ffmpeg

    from auracontent.services.audios.audio import AudioEngine, audio_duration

    # Fausse narration : 3 s de « voix », pause 1 s, 5 s (avec une micro-pause), pause 1 s, 3 s
    fake = tmp_path / "fake.wav"
    tone = "sine=frequency=300:sample_rate=24000:duration={d}"
    gap = "anullsrc=r=24000:cl=mono:d={d}"
    parts = [tone.format(d=3), gap.format(d=1), tone.format(d=2.4), gap.format(d=0.2), tone.format(d=2.4),
             gap.format(d=1), tone.format(d=3)]
    streams = [ffmpeg.input(p, f="lavfi") for p in parts]
    ffmpeg.concat(*streams, v=0, a=1).output(str(fake), ac=1, ar=24000, sample_fmt="s16").run(quiet=True)

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    engine = AudioEngine(AUDIO_CFG, output_dir=str(tmp_path / "out"), min_scene_duration=1.0)
    calls = []
    monkeypatch.setattr(engine, "_gemini_request", lambda prompt, speech, path: (calls.append(1), shutil.copy(fake, path)))

    scenes = [{"id": 1, "text": "x" * 30}, {"id": 2, "text": "x" * 50}, {"id": 3, "text": "x" * 30}]
    asyncio.run(engine.process_script(scenes))

    assert len(calls) == 1  # UNE seule requête TTS pour toute la vidéo
    durations = [audio_duration(s["audio_path"]) for s in scenes]
    assert durations[0] == pytest.approx(3.5, abs=0.3)
    assert durations[1] == pytest.approx(6.0, abs=0.4)
    assert durations[2] == pytest.approx(3.5, abs=0.3)
    assert {s["tts_engine"] for s in scenes} == {"gemini"}


def test_batch_failure_falls_back_to_per_scene(monkeypatch, tmp_path):
    import asyncio
    import wave

    from auracontent.services.audios.audio import AudioEngine

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    engine = AudioEngine(AUDIO_CFG, output_dir=str(tmp_path), min_scene_duration=1.0)

    def boom(scenes):
        raise RuntimeError("découpage aux silences impossible")

    def fake_scene(text, role, path, scene=None):
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(b"\0\0" * 24000)

    monkeypatch.setattr(engine, "_gemini_batch", boom)
    monkeypatch.setattr(engine, "_gemini", fake_scene)
    scenes = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    asyncio.run(engine.process_script(scenes))
    assert all(s["tts_engine"] == "gemini" and s["audio_path"].endswith(".wav") for s in scenes)


# ---------- Direction artistique ----------
def test_srt_never_straddles_two_clauses(tmp_path):
    path = generate_grouped_srt("Ils restent divisés, et le trésor a disparu en 1875.", 5, str(tmp_path / "a.srt"),
                                uppercase=True, highlight_color="#FFD400")
    captions = [b.splitlines()[2] for b in open(path, encoding="utf-8").read().strip().split("\n\n")]
    assert captions[0] == "ILS RESTENT DIVISÉS"                      # coupe à la virgule, virgule masquée
    assert not any(c.endswith((" ET", " LE", " EN")) for c in captions)  # pas de mot-outil en fin de légende
    assert any('<font color="#FFD400">1875</font>' in c for c in captions)


def test_headline_splits_statement_and_punchline():
    from auracontent.services.video.composer import Composer

    composer = Composer({"headline_persistent": True}, channel_name="mystery")
    lines, accent = composer._headline_lines("Mont-Saint-Michel | 70 ans de prison")
    assert lines == ["MONT-SAINT-MICHEL", "70 ANS DE PRISON"] and accent == 1
    assert composer._headline_lines("") == ([], 0)


# ---------- Carte animée (lot 1) ----------
MAP_PLACES = [("Reims", 49.2583, 4.0317, "Grand Est", True),
              ("Ajaccio", 41.9192, 8.7386, "Corse", True),            # au fond du golfe : région la plus proche
              ("Saint-Denis", -20.8789, 55.4481, "Réunion", False)]


def test_wikipedia_page_keeps_primary_coordinates(monkeypatch):
    from auracontent.services.scraping import wikipedia

    page = {"title": "Trésor de Gourdon", "fullurl": "https://fr.wikipedia.org/wiki/T", "extract": "x" * 2000,
            "coordinates": [{"lat": 10.0, "lon": 20.0, "globe": "moon"}, {"lat": 46.64, "lon": 4.45, "globe": "earth"}]}
    monkeypatch.setattr(wikipedia, "_get", lambda params, retries=2: {"query": {"pages": [page]}})
    assert wikipedia._fetch_page("Trésor de Gourdon")["coordinates"] == {"lat": 46.64, "lon": 4.45}

    page.pop("coordinates")
    assert wikipedia._fetch_page("Trésor de Gourdon")["coordinates"] is None


@pytest.mark.parametrize("name, lat, lon, region, metro", MAP_PLACES)
def test_map_finds_region_and_framing(name, lat, lon, region, metro):
    from auracontent.services.visuals.map_scene import find_region, is_metropolitan

    assert find_region(lat, lon)["properties"]["name"] == region
    assert is_metropolitan(lat, lon) is metro


@pytest.mark.parametrize("name, lat, lon, region, metro", MAP_PLACES)
def test_map_point_lands_on_place(name, lat, lon, region, metro):
    """À la fin du zoom, le lieu est sous le point, dans la zone sûre (x < 930, 200 < y < 1440)."""
    from auracontent.services.visuals.map_scene import ACCENT, MapRenderer

    renderer = MapRenderer(lat, lon, name)
    x, y = renderer.to_screen(renderer.end, lon, lat)
    assert (round(x), round(y)) == renderer.ANCHOR
    assert x < 930 and 200 < y < 1440
    img = renderer.render_frame(5.9, 6.0)
    assert img.size == (1080, 1920)
    r, g, b = img.getpixel((round(x), round(y)))
    assert abs(r - ACCENT[0]) < 20 and abs(g - ACCENT[1]) < 20 and b < 60      # point jaune
    lx0, ly0, lx1, ly1 = renderer._label_box(renderer.label_below)
    assert lx1 <= 930 and ly0 > 610 and ly1 < 1200                            # entre bandeau et sous-titres


def test_map_target_disabled_by_default():
    from auracontent.services.visuals.map_scene import map_settings, pick_map_target

    script = {"pick": {"lat": 49.25, "lon": 4.03}}
    scenes = [{"id": 1, "location_name": "Reims"}, {"id": 2}]
    assert map_settings({})["enabled"] is False                  # finance, kids : pas de section map_scene
    assert pick_map_target(script, scenes, map_settings({})) is None


def test_map_target_prefers_wikipedia_then_geocodes():
    from auracontent.services.visuals.map_scene import map_settings, pick_map_target

    cfg = map_settings({"map_scene": {"enabled": True}})
    scenes = [{"id": 1, "location_name": "Cathédrale de Reims", "location_country": "France"},
              {"id": 2, "location_name": "Reims"}, {"id": 3, "role": "cta", "reuse_previous_asset": True}]
    calls = []

    def geocoder(query):
        calls.append(query)
        return 49.25, 4.03

    target = pick_map_target({"pick": {"lat": 48.85, "lon": 2.35}}, scenes, cfg, geocoder=geocoder)
    # sans page source (ancien script) : lieu de la scène 1, jamais celui de la scène qui reçoit la carte
    assert (target.index, target.lat, target.source, target.label) == (1, 48.85, "wikipedia", "Cathédrale de Reims")
    assert calls == []                                              # coordonnées Wikipédia : pas de réseau

    target = pick_map_target({"pick": {}}, scenes, cfg, geocoder=geocoder)
    assert (target.lat, target.source) == (49.25, "nominatim")
    assert calls == ["Cathédrale de Reims, France"]                 # lieu de la scène 1

    assert pick_map_target({"pick": {}}, scenes, cfg, geocoder=lambda q: None) is None
    cta = map_settings({"map_scene": {"enabled": True, "scene_index": 2}})
    assert pick_map_target({"pick": {"lat": 48.85, "lon": 2.35}}, scenes, cta) is None   # jamais sur la scène CTA


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_map_clip_has_exact_duration_and_format(tmp_path):
    import ffmpeg

    from auracontent.services.visuals.map_scene import MapTarget, frame_count, render_map_scene

    out = tmp_path / "map.mp4"
    assert render_map_scene(MapTarget(1, 49.2583, 4.0317, "Reims", "test"), 0.5, str(out), fps=30)
    probe = ffmpeg.probe(str(out), count_frames=None)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"], video["r_frame_rate"]) == (1080, 1920, "30/1")
    assert int(video["nb_read_frames"]) == frame_count(0.5, 30) == 15
    assert not any(s["codec_type"] == "audio" for s in probe["streams"])


def test_map_render_failure_keeps_usual_visual(monkeypatch, tmp_path):
    """Carte impossible -> la scène reçoit son visuel habituel ; carte réussie -> le .mp4 de la carte."""
    from auracontent.core import pipeline as pl
    from auracontent.services.visuals.map_scene import MapTarget

    class FakeChannel:
        name = "mystery"

        def section(self, key):
            return {"visuals": {"map_scene": {"enabled": True}}}.get(key, {})

    vp = pl.VideoPipeline.__new__(pl.VideoPipeline)
    vp.channel, vp.name, vp.work_dir = FakeChannel(), "mystery", tmp_path
    fetched = []
    vp._fetch_scene_asset = lambda scene, **k: fetched.append(scene["id"]) or "usual.jpg"
    scenes = [{"id": 1, "duration": 3.0}, {"id": 2, "duration": 4.0}]
    target = MapTarget(1, 49.25, 4.03, "Reims", "wikipedia")

    monkeypatch.setattr(pl, "render_map_scene", lambda *a, **k: False)
    assets = ["a.jpg", None]
    vp._render_map(target, scenes, assets)
    assert assets == ["a.jpg", "usual.jpg"] and fetched == [2] and "map" not in scenes[1]

    durations = []
    monkeypatch.setattr(pl, "render_map_scene", lambda t, d, out, **k: durations.append(d) or True)
    assets = ["a.jpg", None]
    vp._render_map(target, scenes, assets)
    assert assets[1].endswith("scene_map_2.mp4") and durations == [4.0]   # durée réelle de la voix
    assert scenes[1]["map"]["source"] == "wikipedia"

    # la scène CTA qui suit reprend le visuel de la scène 2 (la carte), pas l'emplacement vide
    scenes.append({"id": 3, "duration": 2.0, "role": "cta", "reuse_previous_asset": True})
    assets = ["a.jpg", None, None]
    vp._render_map(target, scenes, assets)
    assert assets[2] == assets[1] and assets[1].endswith("scene_map_2.mp4")


def test_fetch_assets_leaves_map_scene_for_later(monkeypatch, tmp_path):
    from auracontent.core import pipeline as pl

    class FakeManager:
        def __init__(self, cfg):
            pass

    class FakeChannel:
        name = "mystery"

        def section(self, key):
            return {}

    monkeypatch.setattr(pl, "AssetManager", FakeManager)
    vp = pl.VideoPipeline.__new__(pl.VideoPipeline)
    vp.channel, vp.name, vp.work_dir = FakeChannel(), "mystery", tmp_path
    vp._fetch_scene_asset = lambda scene, **k: f"scene{scene['id']}.jpg"
    scenes = [{"id": 1}, {"id": 2, "location_name": "Reims"}, {"id": 3}]
    assert vp._fetch_assets(scenes, skip={1}) == ["scene1.jpg", None, "scene3.jpg"]


def test_geocode_is_cached_and_never_repeated(monkeypatch, tmp_path):
    """Politique Nominatim : résultats en cache, la même requête n'est jamais renvoyée (même un échec)."""
    from auracontent.services.visuals import map_scene

    monkeypatch.setattr(map_scene, "GEOCODE_CACHE", tmp_path / "geocode_cache.json")
    monkeypatch.setattr(map_scene, "NOMINATIM_MIN_INTERVAL", 0)
    calls = []

    class Resp:
        def __init__(self, hits):
            self.hits = hits

        def raise_for_status(self):
            pass

        def json(self):
            return self.hits

    def fake_get(url, params, headers, timeout):
        calls.append((params["q"], headers["User-Agent"]))
        return Resp([{"lat": "49.2536", "lon": "4.0339"}] if "Reims" in params["q"] else [])

    monkeypatch.setattr(map_scene.requests, "get", fake_get)
    assert map_scene.geocode_nominatim("Cathédrale de Reims", user_agent="test/1.0") == (49.2536, 4.0339)
    assert map_scene.geocode_nominatim("  cathédrale de  REIMS ", user_agent="test/1.0") == (49.2536, 4.0339)
    assert map_scene.geocode_nominatim("Lieu introuvable", user_agent="test/1.0") is None
    assert map_scene.geocode_nominatim("Lieu introuvable", user_agent="test/1.0") is None
    assert [q for q, _ in calls] == ["Cathédrale de Reims", "Lieu introuvable"]
    assert all(agent == "test/1.0" for _, agent in calls)


def test_map_credit_mentions_openstreetmap_only_when_geocoded():
    from auracontent.channels.base_channel import load_channel_config
    from auracontent.services.video.composer import Composer

    composer = Composer(load_channel_config("mystery")["video_composer"], channel_name="mystery")
    assert composer._resolve_source_credit("temp/mystery/assets/scene_map_2.mp4") == "Carte : Natural Earth"
    assert "OpenStreetMap" in composer._resolve_source_credit("temp/mystery/assets/scene_map_osm_2.mp4")


# ---------- Nouveau nom « C’est arrivé ici » (lot 10) ----------
def test_caption_drops_old_name_and_adds_brand_hashtag():
    from auracontent.services.publishing.caption import apply_hashtag_rules

    banned = ["#minutemystere"]
    caption = "🧠 Un trésor perdu !\nVous le saviez ? #histoire #MinuteMystère #france"
    assert apply_hashtag_rules(caption, "#cestarriveici", banned) == (
        "🧠 Un trésor perdu !\nVous le saviez ? #histoire #france #cestarriveici")
    assert apply_hashtag_rules("Légende #CestArriveIci", "#cestarriveici", banned) == "Légende #CestArriveIci"
    assert apply_hashtag_rules("Légende #MinuteMystère", None, ()) == "Légende #MinuteMystère"   # autres chaînes


def test_caption_fallback_follows_brand_rules():
    from auracontent.services.publishing.caption import generate_caption

    class BrokenLLM:
        def call_with_fallback(self, *a, **k):
            raise RuntimeError("quota")

    caption = generate_caption(BrokenLLM(), "Le trésor de Gourdon", [], "#histoire #MinuteMystère",
                               brand_hashtag="#cestarriveici", banned_hashtags=["#minutemystere"])
    assert caption == "Le trésor de Gourdon #histoire #cestarriveici"


def test_mystery_branding_uses_new_name():
    from PIL import Image

    from auracontent.channels.base_channel import load_channel_config
    from auracontent.services.video.composer import Composer

    cfg = load_channel_config("mystery")
    composer = Composer(cfg["video_composer"], channel_name="mystery")
    assert composer.watermark_path.endswith("cest_arrive_ici_watermark.png")
    with Image.open(composer.watermark_path) as logo:
        assert logo.size == (541, 160) and logo.mode == "RGBA"
        bottom = composer.watermark_y + composer.watermark_width * logo.height / logo.width
    assert (composer.watermark_width, composer.watermark_opacity) == (190, 0.85)
    assert bottom < composer.headline_y                      # le filigrane ne touche pas le bandeau-titre
    publish = cfg["publish"]
    assert publish["fallback_hashtags"] == "#histoire #patrimoine #france #cestarriveici"
    assert publish["hf_folder"] == "videos" and publish["zernio_key_env"] == "ZERNIO_API_KEY_MYSTERY"


def test_no_old_channel_name_left():
    """Critère du lot 10 : l'ancien nom de la chaîne n'apparaît plus (hors historique git et CHANGELOG)."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    old_name = "minute" + r"[\s_]" + "myst"           # écrit en morceaux : ce test ne doit pas se trouver lui-même
    pattern = re.compile(old_name + "|" + "mist" + "[eè]re", re.IGNORECASE)
    suffixes = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".sh"}
    hits = []
    for folder in ("auracontent", "config", "scripts", "tests", ".github", "assets"):
        for path in (root / folder).rglob("*"):
            if path.is_file() and (path.suffix in suffixes or path.suffix == ".png") and path != Path(__file__):
                if pattern.search(path.name) or (path.suffix in suffixes and path.stat().st_size < 3_000_000
                                                 and pattern.search(path.read_text(encoding="utf-8", errors="ignore"))):
                    hits.append(str(path.relative_to(root)))
    for name in ("README.md", "CONTRIBUTING.md", ".env.example", "pyproject.toml", "constants.py"):
        path = root / name
        if path.exists() and pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            hits.append(name)
    assert hits == []


def test_map_tagline_is_optional_and_stays_in_safe_zone():
    from auracontent.services.visuals.map_scene import MapRenderer

    plain = MapRenderer(49.2583, 4.0317, "Reims")
    assert plain.tagline == "" and all(not accent for _, _, accent in plain._label_lines())
    for lat, lon, name in ((49.2583, 4.0317, "Reims"), (-20.8789, 55.4481, "Saint-Denis")):
        tagged = MapRenderer(lat, lon, name, {"tagline": "C’est arrivé ici"})
        assert tagged._label_lines()[-1][0] == "C’EST ARRIVÉ ICI" and tagged._label_lines()[-1][2]
        x0, y0, x1, y1 = tagged._label_box(tagged.label_below)
        assert x0 >= 40 and x1 <= 930 and y0 > 610 and y1 < 1200   # entre bandeau et sous-titres


# ---------- Qualité du contenu (fix/qualite-contenu) ----------
def _info(width=3000, height=2000, mime="image/jpeg", license="CC BY-SA 4.0", nonfree="false", desc=""):
    return {"mime": mime, "width": width, "height": height, "url": "https://x/orig.jpg",
            "thumburl": "https://x/thumb.jpg",
            "extmetadata": {"LicenseShortName": {"value": license}, "NonFree": {"value": nonfree},
                            "ImageDescription": {"value": desc}}}


def test_article_images_stop_before_navboxes_and_decode_names():
    from auracontent.services.visuals.subject_images import article_file_titles

    html = ('<a href="/wiki/Fichier:Chaux-les-Passavant,_l%27abbaye.jpg">x</a>'
            '<a href="/wiki/Fichier:Cour_et_b%C3%A2timents.jpg">x</a>'
            '<a href="/wiki/Fichier:Cour_et_b%C3%A2timents.jpg">doublon</a>'
            '<div class="navbox collapsible"><a href="/wiki/Fichier:Notre-Dame_de_Paris.jpg">hors sujet</a></div>')
    assert article_file_titles(html) == ["File:Chaux-les-Passavant, l'abbaye.jpg", "File:Cour et bâtiments.jpg"]


@pytest.mark.parametrize("title, info, ok", [
    ("File:Façade.jpg", _info(), True),
    ("File:Façade.jpg", _info(license="Public domain"), True),
    ("File:Façade.jpg", _info(width=999), False),                        # moins de 1000 px de large
    ("File:Façade.jpg", _info(license="CC BY-NC-SA 4.0"), False),        # pas d'usage commercial
    ("File:Façade.jpg", _info(license="CC BY-ND 2.0"), False),           # pas de recadrage
    ("File:Façade.jpg", _info(license=""), False),                       # licence inconnue
    ("File:Façade.jpg", _info(nonfree="true"), False),
    ("File:Façade.svg", _info(mime="image/svg+xml"), False),
    ("File:Logo monument historique.jpg", _info(), False),
    ("File:Blason de Reims.png", _info(mime="image/png"), False),
    ("File:Doubs department location map.jpg", _info(), False),
    ("File:Pictogramme église.png", _info(mime="image/png"), False),
])
def test_usable_image_rules(title, info, ok):
    from auracontent.services.visuals.subject_images import usable

    assert usable(info, title) is ok


def test_homonyms_need_commune_or_department(monkeypatch):
    """Catégorie devinée (pas de P373) et recherche de documents : la description doit nommer le lieu."""
    from auracontent.services.visuals import subject_images as si

    good = si.SubjectImage("A.jpg", "File:A.jpg", "u", 3000, 2000, "commons", "Abbaye vue de Chaux-lès-Passavant")
    other = si.SubjectImage("B.jpg", "File:B.jpg", "u", 3000, 2000, "commons", "Abbaye de la Grâce-Dieu, Gironde")
    monkeypatch.setattr(si, "article_images", lambda *a, **k: [])
    monkeypatch.setattr(si, "category_images", lambda *a, **k: [good, other])
    terms = ["Chaux-lès-Passavant", "Doubs"]
    guessed = si.collect_subject_images("Abbaye X (Doubs)", None, {"documents": False}, terms)
    assert [i.key for i in guessed] == ["A.jpg"]
    trusted = si.collect_subject_images("Abbaye X (Doubs)", "Abbaye X", {"documents": False}, terms)
    assert [i.key for i in trusted] == ["A.jpg", "B.jpg"]                  # catégorie donnée par Wikidata
    assert si.mentions_place("CHAUX LES PASSAVANT, vue", terms)          # casse, accents et tirets ignorés
    assert si.place_query(terms) == '("Chaux-lès-Passavant" OR "Doubs")'
    assert si.document_images("Abbaye X", []) == []                       # sans lieu connu : aucune recherche


def test_pool_limits_each_file_to_two_uses_never_twice_in_a_row():
    from auracontent.services.visuals.subject_images import SubjectImage, SubjectImagePool

    imgs = [SubjectImage(k, k, k, 3000, 2000, o) for k, o in (("doc", "document"), ("a", "article"), ("c", "commons"))]
    pool = SubjectImagePool(imgs, max_uses=2)
    assert [i.key for i in pool.candidates()] == ["a", "c", "doc"]        # ordre de priorité (a), (b), (c)
    for key in ("a", "c", "doc"):
        pool.record(key)
    assert pool.candidates() == []                                         # plus aucun fichier neuf
    assert [i.key for i in pool.candidates(avoid="doc", first_use_only=False)] == ["a", "c"]
    for key in ("a", "c", "doc"):
        pool.record(key)
    assert pool.candidates(first_use_only=False) == []                     # 2 emplois chacun au plus


def _fake_manager(monkeypatch, tmp_path, images, europeana=False, ai=True, video=True):
    from auracontent.services.visuals import asset_manager as am
    from auracontent.services.visuals.subject_images import SubjectImage

    monkeypatch.setattr(am, "load_history", lambda: {})

    def fake_download(url, path, **k):
        open(path, "wb").write(url.encode() * 50)
        return True

    monkeypatch.setattr(am, "download_file", fake_download)
    m = am.AssetManager({"subject_images": {"enabled": True}})
    m.europeana.get_image = lambda q, path, accept=None: europeana and (accept is None or accept(
        m.europeana.last_text)) and fake_download("europeana", path)
    m.europeana.last_text, m.europeana.last_url = "Gravure, Chaux-lès-Passavant", "europeana"
    m._generate_ai = lambda prompt, path: ai and fake_download(f"ai{len(m.tally)}{m.tally['ai']}", path)
    m.videos.fetch_background = lambda q, path: video and fake_download(f"video{m.tally['video']}", path)
    m.prepare_subject({"source_title": "Abbaye X (Doubs)", "commune": "Chaux-lès-Passavant", "department": "Doubs",
                       "region": "Bourgogne-Franche-Comté"}, era="late 18th century",
                      images=[SubjectImage(k, f"File:{k}", f"https://img/{k}", 3000, 2000, o) for k, o in images])
    return m


def test_every_scene_follows_source_priority(monkeypatch, tmp_path):
    """(a)/(b) neufs -> Europeana -> second emploi -> image générée -> vidéo, pour TOUTES les scènes."""
    m = _fake_manager(monkeypatch, tmp_path, [("a1", "article"), ("c1", "commons")], europeana=True)
    remaining = ["gravure unique"]                       # Europeana : une seule notice pertinente
    m.europeana.get_image = lambda q, path, accept=None: bool(remaining) and open(path, "wb").write(
        remaining.pop().encode() * 99) > 0
    out = str(tmp_path / "x")
    generic = {"id": 1, "scene_type": "generic", "stock_search": "old iron key", "image_prompt": "iron key"}
    sources = []
    for _ in range(6):
        ok, source = m.get_scene_asset(generic, out)
        sources.append((source, m.last_origin))
    assert sources[:2] == [("wiki", "article"), ("wiki", "commons")]       # même une scène 'generic'
    assert sources[2] == ("europeana", "europeana")
    assert sources[3][0] == "wiki" and sources[4][0] == "wiki"            # second emploi des photos du lieu
    assert sources[5] == ("ai", "ai")                                      # Pexels n'est jamais venu avant
    assert "Wikipédia 2, Commons 2, Europeana 1, image générée 1" == m.tally_report()


def test_stock_video_only_as_last_resort(monkeypatch, tmp_path):
    m = _fake_manager(monkeypatch, tmp_path, [], ai=False)
    ok, source = m.get_scene_asset({"id": 1, "stock_search": "candle"}, str(tmp_path / "x"))
    assert (ok, source) == (True, "video")


def test_period_prompt_has_era_region_and_no_modern_objects(monkeypatch, tmp_path):
    m = _fake_manager(monkeypatch, tmp_path, [])
    prompt = m.period_prompt({"image_prompt": "monks carrying stones", "event_context": "monks leave"})
    assert "late 18th century" in prompt and "Chaux-lès-Passavant, Bourgogne-Franche-Comté, France" in prompt
    assert "no modern objects" in prompt and prompt.startswith("monks carrying stones")
    assert m.archive_query({"scene_type": "generic"}) == "Abbaye X Chaux-lès-Passavant"
    assert m.archive_query({"scene_type": "specific", "location_name": "Cloître"}) == "Cloître Chaux-lès-Passavant"


def test_archive_without_place_is_rejected_as_homonym(monkeypatch, tmp_path):
    m = _fake_manager(monkeypatch, tmp_path, [], europeana=True, ai=True)
    m.europeana.last_text = "Abbaye X, Gironde"
    ok, source = m.get_scene_asset({"id": 1}, str(tmp_path / "x"))
    assert source == "ai"


def test_same_file_never_more_than_twice(monkeypatch, tmp_path):
    """La limite porte sur le contenu du fichier, quelle que soit la source qui le renvoie."""
    m = _fake_manager(monkeypatch, tmp_path, [], ai=True, video=False)
    m._generate_ai = lambda prompt, path: open(path, "wb").write(b"meme image" * 100) > 0
    out = str(tmp_path / "x")
    assert m._generate_ai("p", out) and m._accept(out, "ai")
    assert not m._accept(out, "ai")                         # jamais deux fois de suite
    m.last_key = None
    assert m._accept(out, "ai")
    m.last_key = None
    assert not m._accept(out, "ai")                         # troisième emploi refusé


@pytest.mark.parametrize("duration, index, expected", [(7.0, 0, 3), (3.0, 0, 1), (7.0, 1, 2), (8.5, 2, 3), (4.0, 3, 1)])
def test_shot_count_follows_voice_duration(duration, index, expected):
    from auracontent.core.pipeline import VideoPipeline

    cfg = {"max_seconds": 4.0, "first_scene_max_seconds": 3.0, "words_per_second": 2.4, "max_shots_per_scene": 6}
    assert VideoPipeline.shot_count({"duration": duration}, index, cfg) == expected


def test_no_shot_longer_than_limit():
    from auracontent.services.video.composer import Composer

    for duration, limit in ((7.3, 3.0), (8.0, 4.0), (11.9, 4.0), (2.0, 3.0)):
        frames = Composer.shot_frames(duration, limit, 30)
        assert sum(frames) == math.ceil(duration * 30 - 1e-6)
        assert max(frames) / 30 <= limit + 1e-9 and max(frames) - min(frames) <= 1


def test_fetch_assets_gives_each_scene_enough_shots(tmp_path):
    from auracontent.core import pipeline as pl

    class FakeChannel:
        name = "mystery"

        def section(self, key):
            return {"video_composer": {"shots": {"enabled": True}}}.get(key, {})

    class FakeManager:
        def __init__(self, cfg):
            pass

    vp = pl.VideoPipeline.__new__(pl.VideoPipeline)
    vp.channel, vp.name, vp.work_dir = FakeChannel(), "mystery", tmp_path
    pl_manager, pl.AssetManager = pl.AssetManager, FakeManager
    try:
        vp._fetch_scene_asset = lambda scene, suffix="", **k: f"s{scene['id']}{suffix}.jpg"
        scenes = [{"id": 1, "duration": 7.0}, {"id": 2, "duration": 7.0}, {"id": 3, "duration": 5.0},
                  {"id": 4, "duration": 2.0, "role": "cta", "reuse_previous_asset": True}]
        assets = vp._fetch_assets(scenes, skip={1})
    finally:
        pl.AssetManager = pl_manager
    assert assets[0] == {"shots": ["s1a.jpg", "s1b.jpg", "s1c.jpg"], "max_shot": 3.0}
    assert assets[1] is None                                               # carte, fabriquée après la voix
    assert assets[2] == {"shots": ["s3a.jpg", "s3b.jpg"], "max_shot": 4.0}
    assert assets[3] == {"shots": ["s3b.jpg"], "max_shot": 4.0}            # CTA : dernier plan seulement


def test_report_warns_when_stock_video_used_in_several_scenes(capsys):
    from auracontent.core import pipeline as pl

    class FakeChannel:
        def section(self, key):
            return {}

    vp = pl.VideoPipeline.__new__(pl.VideoPipeline)
    vp.channel = FakeChannel()
    vp._scene_origins = {0: ["article", "commons"], 2: ["video"], 3: ["video", "ai"], 1: ["map"]}
    lines = vp.report_sources()
    assert any("Pexels / Pixabay utilisé dans 2 scènes (scènes 3, 4)" in line for line in lines)
    vp._scene_origins = {0: ["video"], 1: ["article"]}
    assert not any("⚠️" in line for line in vp.report_sources())


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_landscape_photo_fills_screen_and_pans(tmp_path):
    """Paysage : plein écran (aucun fond flou) et déplacement visible entre la première et la dernière image."""
    import ffmpeg
    from PIL import Image

    from auracontent.services.video.composer import Composer

    image = tmp_path / "wide.jpg"
    img = Image.new("RGB", (1800, 900))
    img.putdata([(x * 255 // 1800, 120, 255 - x * 255 // 1800) for y in range(900) for x in range(1800)])
    img.save(image, quality=95)
    voice = tmp_path / "voice.wav"
    ffmpeg.input("sine=duration=2", f="lavfi").output(str(voice), ac=2, ar=48000).run(quiet=True)

    composer = Composer({"video_width": 360, "video_height": 640, "fps": 15, "image_full_screen": True,
                         "pan_speed": 0.2}, channel_name="mystery")
    composer.temp_dir = str(tmp_path)
    out = composer.process_scene({"id": "p", "audio_path": str(voice), "duration": 2.0},
                                 {"shots": [str(image)], "max_shot": 4.0})
    frames = []
    for t in (0.05, 1.9):
        png = tmp_path / f"f{t}.png"
        ffmpeg.input(out, ss=t).output(str(png), vframes=1).run(quiet=True)
        frames.append(Image.open(png).convert("RGB"))
    first = frames[0]
    top, middle = first.getpixel((180, 20)), first.getpixel((180, 320))
    assert all(abs(a - b) < 12 for a, b in zip(top, middle))   # même colonne de l'image : pas de bande floue
    assert abs(frames[0].getpixel((180, 320))[0] - frames[1].getpixel((180, 320))[0]) > 8   # ça bouge


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_multi_shot_scene_keeps_duration(tmp_path):
    import ffmpeg
    from PIL import Image

    from auracontent.services.video.composer import Composer

    shots = []
    for i, color in enumerate(((200, 40, 40), (40, 200, 40), (40, 40, 200))):
        path = tmp_path / f"s{i}.jpg"
        Image.new("RGB", (900, 1600), color).save(path)
        shots.append(str(path))
    voice = tmp_path / "voice.wav"
    ffmpeg.input("sine=duration=7", f="lavfi").output(str(voice), ac=2, ar=48000).run(quiet=True)
    composer = Composer({"video_width": 180, "video_height": 320, "fps": 10, "image_full_screen": True},
                        channel_name="mystery")
    composer.temp_dir = str(tmp_path)
    out = composer.process_scene({"id": "m", "audio_path": str(voice), "duration": 7.0},
                                 {"shots": shots, "max_shot": 3.0})
    assert 6.9 <= float(ffmpeg.probe(out)["format"]["duration"]) <= 7.6
    colors = []
    for t in (1.0, 3.5, 6.0):                                   # trois plans de 2,33 s
        png = tmp_path / f"c{t}.png"
        ffmpeg.input(out, ss=t).output(str(png), vframes=1).run(quiet=True)
        colors.append(max(range(3), key=lambda c: Image.open(png).convert("RGB").getpixel((90, 100))[c]))
    assert colors == [0, 1, 2]


def test_map_label_is_subject_or_commune_never_geocoder():
    from auracontent.services.visuals.map_scene import map_labels, map_settings, pick_map_target

    cfg = map_settings({"map_scene": {"enabled": True}})
    scenes = [{"id": 1, "location_name": "Cloître"}, {"id": 2, "location_name": "Cascade voisine"}]
    pick = {"source_title": "Abbaye de la Grâce-Dieu (Doubs)", "commune": "Chaux-lès-Passavant",
            "region": "Bourgogne-Franche-Comté"}
    assert map_labels({"pick": pick}, scenes, cfg) == ("Abbaye de la Grâce-Dieu", "Chaux-lès-Passavant")
    long_pick = {**pick, "source_title": "Trésor de la cathédrale Saint-Étienne de Sens"}
    assert map_labels({"pick": long_pick}, scenes, cfg) == ("Chaux-lès-Passavant", "")
    calls = []
    target = pick_map_target({"pick": pick}, scenes, cfg, geocoder=lambda q: calls.append(q) or (47.24, 6.36))
    assert target.label == "Abbaye de la Grâce-Dieu" and target.source == "nominatim"
    assert calls == ["Abbaye de la Grâce-Dieu, Chaux-lès-Passavant, France"]   # le sujet, pas une autre scène


def test_map_zoom_starts_on_first_frame_and_shows_nearest_big_city():
    from auracontent.services.visuals.map_scene import MapRenderer

    r = MapRenderer(47.2436, 6.3656, "Abbaye de la Grâce-Dieu", sublabel="Chaux-lès-Passavant")
    assert r.camera(1 / 30, 6.0).scale > r.start.scale * 1.01            # aucun plan fixe au début
    assert r.landmark["name"] == "BESANÇON"
    x, y = r.to_screen(r.end, r.landmark["lon"], r.landmark["lat"])
    assert 40 <= x <= 930 and 620 <= y <= 1400
    assert r._label_lines()[1][0] == "CHAUX-LÈS-PASSAVANT"
    assert MapRenderer(49.2583, 4.0317, "Reims").landmark is None        # le sujet EST la grande ville


def _mystery_brain(llm=None):
    from auracontent.channels.base_channel import load_channel_config
    from auracontent.channels.mystery.brain import MysteryBrain

    return MysteryBrain(config=load_channel_config("mystery"), llm=llm)


class _ScriptedLLM:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def call_json_with_retry(self, messages, **k):
        self.calls.append(messages)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_hooks_rated_zero_or_equal_to_title_are_regenerated():
    topic = "Le Mont-Saint-Michel a servi de prison pendant 70 ans"
    llm = _ScriptedLLM(
        {"hooks": [{"text": "Le Mont-Saint-Michel a servi de prison pendant 70 ans", "score": 9},
                   {"text": "Des moines, puis des forçats derrière ces murs", "score": 0}]},
        {"hooks": [{"text": "Pendant 70 ans, on y enfermait des prisonniers"}]},
        {"notes": [{"index": 0, "phrase_complete": True, "verbe": "enfermait", "score": 7}]},   # gros modèle
    )
    hooks = _mystery_brain(llm).generate_hook_variants(topic)
    assert [h["text"] for h in hooks] == ["Pendant 70 ans, on y enfermait des prisonniers"]
    assert hooks[0]["score"] == 7 and len(llm.calls) == 3

    never = _ScriptedLLM(*[{"hooks": [{"text": topic, "score": 8}]}] * 3)
    with pytest.raises(RuntimeError):
        _mystery_brain(never).generate_hook_variants(topic)    # jamais d'accroche par défaut


def test_hook_rating_uses_big_models_and_needs_a_conjugated_verb():
    hooks = [{"text": "Une statue, un saint, un général gaulois"}, {"text": "Ce saint a le visage d'un Gaulois"}]
    llm = _ScriptedLLM({"notes": [{"index": 0, "phrase_complete": False, "verbe": None, "score": 9},
                                  {"index": 1, "phrase_complete": True, "verbe": "a", "score": 6}]})
    captured = {}
    llm.call_json_with_retry = lambda messages, **k: captured.update(k) or llm.answers.pop(0)
    brain = _mystery_brain(llm)
    rated = brain.rate_hooks(hooks, "t")
    assert [h["text"] for h in rated] == ["Ce saint a le visage d'un Gaulois"]
    assert "gpt-oss-20b" not in captured["allowed_models"] and "gpt-oss-120b" in captured["allowed_models"]


def test_dates_and_numbers_must_be_in_the_source():
    from auracontent.channels.mystery.brain import claim_issues, numbers_in, title_format

    source = ("Dans l'église Saint-Sigismond de Larressingle, la statue du saint patron est une réduction de "
              "celle de Vercingétorix. Le monument d'Aimé Millet domine Alise-Sainte-Reine depuis 1865, "
              "au XIXe siècle.")
    assert numbers_in("1 200 moines au XIXe siècle") == {"1200", "19"}
    assert claim_issues("Larressingle : en 1865, le saint prend un visage gaulois", source, "titre") == []
    assert "1450" in claim_issues("En 1450, la statue de Larressingle change de visage", source, "titre")[0]
    assert claim_issues("Trois statues changent de visage", source, "titre")       # « trois » absent
    assert claim_issues("Ce saint a les traits de Vercingétorix", source, "accroche") == []   # nom propre
    assert "ni chiffre" in claim_issues("Ce saint a un drôle de visage", source, "accroche")[0]
    undated = "La statue du saint patron de Larressingle reprend le visage de Vercingétorix."
    formats = {title_format(undated) for _ in range(60)}
    assert not any("année" in f or "chiffre" in f for f in formats)             # jamais sans date ni chiffre


def test_headline_rejects_a_date_absent_from_the_source():
    brain = _mystery_brain(_ScriptedLLM())
    brain._source = {"title": "Larressingle", "extract": "La statue de Larressingle copie Vercingétorix."}
    with pytest.raises(ValueError, match="1450"):
        brain.validate_headline({"on_screen_title": "EN 1450 | LE SAINT GAULOIS", "scenes": []})


def test_veracity_warning_at_the_limit_rejects_the_topic():
    from auracontent.core.brain.base_brain import TopicRejected

    brain = _mystery_brain(_ScriptedLLM(*[{"scenes": [{"id": i, "text": "x"} for i in range(1, 6)]}] * 3))
    brain.target_scene_count = lambda: 5
    brain._validate_script_base = lambda data: None
    brain.validate_script_specifics = lambda data, topic: None
    brain.config["brain"]["scene_count_tolerance"] = 10
    calls = []
    brain.check_script_logic = lambda topic, data: calls.append(1) or [
        brain_module().VeracityIssue("scène 1 : date absente de la source"), "scène 1 : phrase de 14 mots"]
    with pytest.raises(TopicRejected, match="date absente"):
        brain.generate_script("t")
    assert len(calls) == 3                         # pas de nouvel essai avec moins de scènes : sujet abandonné

    style_only = _mystery_brain(_ScriptedLLM(*[{"scenes": [{"id": i, "text": "x"} for i in range(1, 6)]}] * 3))
    style_only.target_scene_count = lambda: 5
    style_only.validate_script_specifics = lambda data, topic: None
    style_only.config["brain"]["scene_count_tolerance"] = 10
    style_only.check_script_logic = lambda topic, data: ["scène 1 : phrase de 14 mots"]
    assert style_only.generate_script("t")["unresolved_issues"]   # le style seul reste un avertissement


def brain_module():
    from auracontent.core.brain import base_brain

    return base_brain


def test_strict_factcheck_unavailable_never_passes():
    brain = _mystery_brain(_ScriptedLLM(RuntimeError("quota")))
    brain._source = {"title": "X", "extract": "y" * 2000}
    with pytest.raises(RuntimeError, match="véracité indisponible"):
        brain._llm_factcheck("t", [{"id": 1, "text": "a"}], {})
    brain = _mystery_brain(_ScriptedLLM({"issues": ["« s'inspire des gravures » : absent de la source"]}))
    brain._source = {"title": "X", "extract": "y" * 2000, "facts": ["f1"], "answer": "r"}
    issues = brain._llm_factcheck("t", [{"id": 1, "text": "a"}], {})
    assert isinstance(issues[0], brain_module().VeracityIssue)


def test_rejected_topic_moves_to_the_next_one(monkeypatch, tmp_path):
    from auracontent.core.brain import global_brain as gb
    from auracontent.core.brain.base_brain import TopicRejected
    from auracontent.utils import pending_topic as pt

    monkeypatch.setattr(pt, "PENDING_PATH", tmp_path / "pending.json")

    class Brain:
        last_pick = {}
        picks = []

        def pick_topic(self):
            n = len(self.picks) + 1
            self.picks.append(n)
            self.last_pick = {"source_title": f"Page {n}", "anecdote_id": str(n)}
            return f"Sujet {n}"

        def generate_hook_variants(self, topic, **k):
            return [{"text": "accroche", "pattern": "chiffre"}]

        def generate_script(self, topic, chosen_hook=None):
            if topic == "Sujet 1":
                raise TopicRejected("fait absent de la source")
            return {"scenes": []}

        def export_state(self):
            return {}

        def restore_state(self, state):
            pass

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
    assert '"topic": "Sujet 2"' in path.read_text(encoding="utf-8")
    titles, ids = pt.load_rejected("mystery")
    assert titles == {"page 1"} and ids == {"1"}                 # plus jamais proposé
    assert pt.load_pending("mystery") is None


def test_facts_extraction_rejects_thin_or_unanswered_topics():
    trial = {"source": {"title": "Larressingle", "extract": "x", "anecdote": "a"}, "focus": ["statue"]}
    thin = _mystery_brain(_ScriptedLLM({"faits": ["f1", "f2", "f3"], "reponse": "r"}))
    assert not thin.extract_facts(dict(trial), "t")              # moins de 4 faits
    unanswered = _mystery_brain(_ScriptedLLM({"faits": ["f1", "f2", "f3", "f4"], "reponse": None}))
    assert not unanswered.extract_facts(dict(trial), "t")       # la source ne répond pas au titre
    ok = _mystery_brain(_ScriptedLLM({"faits": ["f1", "f2", "f3", "f4"], "reponse": "Le sculpteur a réduit..."}))
    t = dict(trial)
    assert ok.extract_facts(t, "t") and t["answer"].startswith("Le sculpteur")


def test_short_material_gives_a_short_video(monkeypatch):
    monkeypatch.delenv("VIDEO_DURATION", raising=False)
    brain = _mystery_brain()
    brain._source = {"facts": ["f"] * 5}
    assert brain.target_scene_count() == 4                       # environ 30 s plutôt que 45
    brain._source = {"facts": ["f"] * 9}
    assert brain.target_scene_count() > 4


def test_script_prompt_carries_anecdote_cited_articles_and_facts():
    brain = _mystery_brain()
    brain._source = {"title": "Larressingle", "extract": "Commune du Gers.", "anecdote": "La statue copie Vercingétorix.",
                     "anecdote_caption": "Statue de saint Sigismond",
                     "related": [{"title": "Monument à Vercingétorix (Millet)", "extract": "Statue de 1865."}],
                     "facts": ["La statue est une réduction de celle d'Aimé Millet."], "answer": "Réduction.",
                     "focus": ["statue de saint Sigismond"]}
    messages = brain.build_script_prompt("t", 3, {"text": "h"})
    user = messages[1]["content"]
    for part in ("La statue copie Vercingétorix", "Statue de saint Sigismond", "Statue de 1865",
                 "réduction de celle d'Aimé Millet", "RÉPONSE", "justifier une image"):
        assert part in user


def test_reveal_sentence_is_located_in_its_scene():
    """Choisie par le code (plus proche de la réponse), jamais par le LLM ; jamais d'échec du script."""
    brain = _mystery_brain()
    brain._source = {"answer": "Le visage du saint est celui de Vercingétorix."}
    data = {"reveal_sentence": "Phrase inventée par le LLM.",
            "scenes": [{"text": "Une statue. Un saint."}, {"text": "Mais regardez bien. Le saint a le visage de "
                                                                    "Vercingétorix."}]}
    brain.locate_reveal(data)
    assert data["scenes"][1]["reveal_sentence"] == "Le saint a le visage de Vercingétorix."
    assert "reveal_sentence" not in data["scenes"][0]


def test_style_flags_long_sentences_and_decorative_adjectives():
    brain = _mystery_brain()
    scenes = [{"id": 1, "text": "Les moines partent en 1791. L'abbaye est vendue."},
              {"id": 2, "text": "Sur ce plateau escarpé, les religieux bâtissent un monastère qui deviendra ensuite "
                                "un lieu de pèlerinage connu."},
              {"id": 3, "text": "Un rôle majeur. Des pentes escarpées."}]
    issues = brain.check_style(scenes)
    assert len(issues) == 3
    assert "scène 2 : phrase de 17 mots" in issues[0] and "escarpé" in issues[1]
    assert "escarpées" in issues[2] and "majeur" in issues[2]


def test_at_least_four_specific_scenes_out_of_six():
    brain = _mystery_brain()
    assert brain.min_specific(6) == 4
    scenes = [{"id": i, "scene_type": "specific" if i < 3 else "generic"} for i in range(6)]
    assert brain.check_specific_count(scenes)
    scenes[3]["scene_type"] = "specific"
    assert brain.check_specific_count(scenes) == []


def test_headline_needs_verb_or_number_and_not_only_places():
    llm = _ScriptedLLM({"verbe": True}, {"verbe": False}, RuntimeError("quota"))
    brain = _mystery_brain(llm)
    brain.last_pick = {"source_title": "Abbaye de la Grâce-Dieu (Doubs)", "commune": "Chaux-lès-Passavant"}
    scenes = [{"location_name": "Abbaye de la Grâce-Dieu"}]
    brain.validate_headline({"on_screen_title": "EN 1791 | L'ABBAYE VIDÉE", "scenes": scenes})   # chiffre : pas de LLM
    assert llm.calls == []
    with pytest.raises(ValueError, match="noms de lieux"):
        brain.validate_headline({"on_screen_title": "GRÂCE-DIEU | CHAUX-LÈS-PASSAVANT", "scenes": scenes})
    brain.validate_headline({"on_screen_title": "L'ABBAYE | VENDUE AUX ENCHÈRES", "scenes": scenes})
    with pytest.raises(ValueError, match="ni verbe ni chiffre"):
        brain.validate_headline({"on_screen_title": "L'ABBAYE | ET SES MOINES", "scenes": scenes})
    with pytest.raises(ValueError):                             # LLM indisponible : rejeté, jamais accepté
        brain.validate_headline({"on_screen_title": "L'ABBAYE | ET SES MOINES", "scenes": scenes})


def test_title_without_reversal_is_rejected():
    source = {"title": "Abbaye X", "extract": "..."}
    assert _mystery_brain(_ScriptedLLM({"retournement": True, "raison": "vendue"})).has_reversal("t", source)[0]
    assert not _mystery_brain(_ScriptedLLM({"retournement": False})).has_reversal("Fondée en 1139", source)[0]
    assert not _mystery_brain(_ScriptedLLM(RuntimeError("quota"))).has_reversal("t", source)[0]


def test_topic_needs_enough_images_and_falls_back_to_best_illustrated(monkeypatch, capsys):
    from auracontent.channels.mystery import brain as mb

    monkeypatch.setattr(mb, "load_global_history", lambda: [])
    monkeypatch.setattr(mb, "load_used_sources", lambda: set())
    monkeypatch.setattr(mb, "load_last_regions", lambda **k: [])
    monkeypatch.setattr(mb, "load_last_categories", lambda **k: [])
    monkeypatch.setattr(mb, "place_details", lambda qid: {"commune": "C"})
    pages = iter([{"title": f"Lieu {i}", "url": "u", "extract": "x"} for i in range(10)])
    counts = {"Lieu 0": 2, "Lieu 1": 4, "Lieu 2": 1}
    brain = _mystery_brain()
    brain._anecdotes = []                                      # « Le saviez-vous ? » vide : repli mots-clés
    brain._find_source = lambda region, category, used: next(pages)
    brain.count_source_images = lambda source, details, focus=None: (counts.get(source["title"], 0),) * 2
    titles = []
    brain._accept_title = lambda trial, used, check_reversal=True: (
        titles.append(trial["source"]["title"]) or f"Titre {trial['source']['title']}")
    assert brain.pick_topic() == "Titre Lieu 1"                # 10 essais, aucun à 5 images : le mieux illustré
    assert titles == ["Lieu 1"] and brain.last_pick["source_images"] == 4
    out = capsys.readouterr().out
    assert "« Lieu 0 » : 2 image(s) exploitable(s) (minimum 5)" in out and "Essai 10/10" in out

    pages = iter([{"title": f"Lieu {i}", "url": "u", "extract": "x"} for i in range(10)])
    counts.update({"Lieu 1": 7})
    assert brain.pick_topic() == "Titre Lieu 1"                # seuil atteint : retenu tout de suite
    assert brain.last_pick["commune"] == "C"


def test_script_uses_only_large_models_and_waits(monkeypatch):
    from auracontent.core.brain import llm_client as lc

    client = lc.LLMClient()
    providers = [{"model": "gemini-3.6-flash"}, {"model": "llama-3.3-70b"}, {"model": "openai/gpt-oss-20b"}]
    monkeypatch.setattr(lc.LLMClient, "providers", property(lambda self: providers))
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    seen, results = [], [RuntimeError("429"), "ok"]

    def fake_call(self, messages, temperature, json_mode, mct, cap, allowed):
        seen.append([p["model"] for p in providers if self.model_allowed(p["model"], allowed)])
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(lc.LLMClient, "_call_providers", fake_call)
    assert client.call_with_fallback([], allowed_models=["gemini", "gpt-oss-120b"], wait_rounds=2) == "ok"
    assert seen == [["gemini-3.6-flash"], ["gemini-3.6-flash"]]          # jamais llama-70b ni gpt-oss-20b
    brain = _mystery_brain()
    routing = brain.script_routing()
    assert "gpt-oss-120b" in routing["allowed_models"] and not lc.LLMClient.model_allowed("openai/gpt-oss-20b",
                                                                                         routing["allowed_models"])


def test_gallica_keeps_public_domain_records_only():
    from auracontent.services.scraping.gallica_provider import parse_records

    xml = """<srw:searchRetrieveResponse xmlns:srw="http://www.loc.gov/zing/srw/"
      xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <srw:records><srw:record><srw:recordData><oai_dc:dc>
        <dc:title>Abbaye de la Grâce-Dieu : gravure</dc:title>
        <dc:identifier>https://gallica.bnf.fr/ark:/12148/btv1b8451602x</dc:identifier>
        <dc:rights>domaine public</dc:rights></oai_dc:dc></srw:recordData></srw:record>
      <srw:record><srw:recordData><oai_dc:dc><dc:title>Photo récente</dc:title>
        <dc:identifier>https://gallica.bnf.fr/ark:/12148/btv1b0000000a</dc:identifier>
        <dc:rights>restricted use</dc:rights></oai_dc:dc></srw:recordData></srw:record></srw:records>
      </srw:searchRetrieveResponse>"""
    records = parse_records(xml)
    assert [(r["ark"], r["public"]) for r in records] == [("ark:/12148/btv1b8451602x", True),
                                                           ("ark:/12148/btv1b0000000a", False)]
    assert parse_records("<html>Cloudflare</html") == []


def test_place_details_walks_up_to_the_department(monkeypatch):
    from auracontent.services.scraping import wikipedia

    def claim(prop, qid):
        return {prop: [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": qid}}}}]}

    entities = {
        "Q1": {"claims": {**claim("P131", "Q2"), "P373": [{"mainsnak": {"datavalue": {"value": "Abbaye X"}}}]}},
        "Q2": {"labels": {"fr": {"value": "Chaux-lès-Passavant"}}, "claims": claim("P131", "Q3")},
        "Q3": {"labels": {"fr": {"value": "canton de Vercel"}}, "claims": {**claim("P31", "Q18524218"),
                                                                           **claim("P131", "Q4")}},
        "Q4": {"labels": {"fr": {"value": "Doubs"}}, "claims": claim("P31", "Q6465")},
    }
    monkeypatch.setattr(wikipedia, "_wikidata", lambda params: {"entities": {params["ids"]: entities[params["ids"]]}})
    assert wikipedia.place_details("Q1") == {"commons_category": "Abbaye X", "commune": "Chaux-lès-Passavant",
                                             "department": "Doubs"}
    assert wikipedia.place_details(None) == {}


# ---------- Quotas LLM (fix/quotas-llm) ----------
class _FakeCompletions:
    def __init__(self, outcomes, calls):
        self.outcomes, self.calls = outcomes, calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _fake_openai(monkeypatch, outcomes):
    from auracontent.core.brain import llm_client as lc

    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": _FakeCompletions(outcomes, calls)})()

    monkeypatch.setattr(lc, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    return calls


def _completion(text, prompt_tokens=10, completion_tokens=5):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


def _client_with(monkeypatch, groups):
    """LLMClient dont les fournisseurs sont fabriqués par _provider_group à partir des vraies specs."""
    from auracontent.core.brain import llm_client as lc

    monkeypatch.setattr(lc, "_discover_models", lambda name, url, key, configured, prefs, limit=3: configured)
    monkeypatch.setattr(lc, "_base_url", lambda name: "https://example.invalid/v1")
    return lc.LLMClient(keys={g: "k" for g in groups}, models=groups, order=list(groups))


def test_groq_gpt_oss_skips_strict_json_and_lowers_reasoning(monkeypatch):
    from auracontent.core.brain import llm_client as lc

    client = _client_with(monkeypatch, {"groq": ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]})
    gpt, llama = client.providers
    assert gpt["json_mode"] is False and llama["json_mode"] is True
    assert gpt["extra"] == {"reasoning_effort": "low", "include_reasoning": False} and llama["extra"] == {}
    assert lc.split_extras(gpt["extra"]) == {"reasoning_effort": "low", "extra_body": {"include_reasoning": False}}

    calls = _fake_openai(monkeypatch, [_completion('```json\n{"ok": true}\n```')])
    assert client.call_json_with_retry([{"role": "user", "content": "x"}], max_completion_tokens=500) == {"ok": True}
    assert "response_format" not in calls[0]                       # jamais le mode JSON strict de Groq
    assert calls[0]["extra_body"] == {"include_reasoning": False}


def test_reasoning_models_get_a_bigger_first_budget(monkeypatch):
    from auracontent.core.brain import llm_client as lc

    client = _client_with(monkeypatch, {"github": ["openai/gpt-4.1-mini"], "openrouter": ["qwen/qwen3-235b"]})
    monkeypatch.setattr(lc, "_discover_free_models", lambda: [])
    calls = _fake_openai(monkeypatch, [_completion("a"), _completion("b")])
    client.call_with_fallback([], max_completion_tokens=1000, allowed_models=["gpt-4.1-mini"])
    client.call_with_fallback([], max_completion_tokens=1000, allowed_models=["qwen3"])
    assert calls[0]["max_completion_tokens"] == 1000                      # modèle sans raisonnement
    assert calls[1]["max_completion_tokens"] == 1000 + lc.REASONING_EXTRA_TOKENS
    assert calls[1]["extra_body"] == {"reasoning": {"effort": "low"}}     # OpenRouter : effort abaissé


def test_rejected_reasoning_parameter_is_dropped(monkeypatch):
    client = _client_with(monkeypatch, {"zhipu": ["glm-4.5-flash"]})
    calls = _fake_openai(monkeypatch, [Exception("Error code: 400 - unknown parameter: thinking"), _completion("ok")])
    assert client.call_with_fallback([]) == "ok"
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}} and "extra_body" not in calls[1]


def test_usage_report_counts_calls_tokens_and_failures(monkeypatch):
    client = _client_with(monkeypatch, {"github": ["openai/gpt-4.1-mini"]})
    _fake_openai(monkeypatch, [Exception("Error code: 500 boom"), _completion("ok", 120, 30)])
    client.call_with_fallback([])
    assert client.usage["GitHub Models"] == {"calls": 2, "failures": 1, "prompt_tokens": 120, "completion_tokens": 30}
    report = "\n".join(client.usage_report())
    assert "GitHub Models" in report and "2 appel(s),   1 échec(s)" in report
    assert "Total : 2 appel(s), dont 1 échoué(s)." in report


@pytest.mark.parametrize("raw, expected", [("OK", "OK"), ('{"choices": [{"message": {"content": "OK"}}]}', "OK"),
                                           (b"OK", "OK")])
def test_github_models_plain_text_answer_is_accepted(raw, expected):
    """Régression du diagnostic : « Réponse inattendue du LLM : OK » alors que le modèle avait répondu."""
    from auracontent.core.brain.llm_client import LLMClient

    assert LLMClient._extract_content(raw) == expected


def test_simple_tasks_fall_back_without_script_models(monkeypatch):
    client = _client_with(monkeypatch, {"gemini": ["gemini-3.6-flash"], "groq": ["openai/gpt-oss-20b"],
                                        "cloudflare": ["@cf/meta/llama-3.3-70b-instruct-fp8-fast"]})
    calls = _fake_openai(monkeypatch, [Exception("Error code: 402 more credits"), _completion("ok")])
    out = client.call_with_fallback([], allowed_models=["gpt-oss-20b"], fallback_excluded=["gemini", "gpt-oss-120b"])
    assert out == "ok" and [c["model"] for c in calls] == ["openai/gpt-oss-20b",
                                                           "@cf/meta/llama-3.3-70b-instruct-fp8-fast"]


def test_mystery_routes_simple_tasks_to_small_models():
    brain = _mystery_brain(_ScriptedLLM({"retournement": True}))
    seen = []
    brain.llm.call_json_with_retry = lambda messages, **k: seen.append(k) or {"retournement": True}
    brain.has_reversal("t", {"title": "x"})
    assert "gpt-oss-20b" in seen[0]["allowed_models"] and "gemini" not in seen[0]["allowed_models"]
    assert "gemini" in seen[0]["fallback_excluded"]                    # Gemini réservé au script
    assert "gemini" in brain.script_routing()["allowed_models"]


def test_no_removed_provider_left():
    """Mistral et Cerebras retirés : code, configuration, workflows, documentation, listes de modèles."""
    import re
    from pathlib import Path

    from auracontent.channels.base_channel import load_channel_config
    from auracontent.core.brain import llm_client

    root = Path(__file__).resolve().parent.parent
    pattern = re.compile("mis" + "tral|cere" + "bras", re.IGNORECASE)   # en morceaux : ce test ne se trouve pas lui-même
    hits = []
    for folder in ("auracontent", "config", "scripts", ".github"):
        for path in (root / folder).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".toml"} and \
                    pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                hits.append(str(path.relative_to(root)))
    for name in ("README.md", "CONTRIBUTING.md", ".env.example", "pyproject.toml"):
        if pattern.search((root / name).read_text(encoding="utf-8")):
            hits.append(name)
    assert hits == []
    assert not {"mistral", "cerebras"} & set(llm_client.PROVIDER_SPECS)
    brain = load_channel_config("mystery")["brain"]
    assert not any(pattern.search(m) for m in brain["script_models"] + brain["simple_models"])


def test_script_accepted_one_scene_off_target():
    from auracontent.core.brain.base_brain import BaseBrain

    class Brain(BaseBrain):
        def build_script_prompt(self, *a):
            return []

    def scenes(n):
        return {"title": "t", "scenes": [{"id": 9, "text": "x"} for _ in range(n)]}

    llm = _ScriptedLLM(scenes(5))
    data = Brain(config={"brain": {"scene_count_tolerance": 1}}, llm=llm)._generate_script_attempt("t", 6, None, 0)
    assert [s["id"] for s in data["scenes"]] == [1, 2, 3, 4, 5]           # renumérotées
    with pytest.raises(ValueError):
        Brain(config={"brain": {"scene_count_tolerance": 1}}, llm=_ScriptedLLM(scenes(4)))._generate_script_attempt(
            "t", 6, None, 0)
    with pytest.raises(ValueError):                                       # autres chaînes : exact
        Brain(config={"brain": {}}, llm=_ScriptedLLM(scenes(5)))._generate_script_attempt("t", 6, None, 0)


def test_pending_topic_survives_script_failure(monkeypatch, tmp_path):
    from auracontent.core.brain import global_brain as gb
    from auracontent.utils import pending_topic

    monkeypatch.setattr(pending_topic, "PENDING_PATH", tmp_path / "pending_topics.json")
    saved = []
    monkeypatch.setattr(gb, "save_global_history", lambda *a, **k: saved.append(a))
    monkeypatch.setattr(gb, "load_hook_history", lambda channel: [])

    class FakeBrain:
        def __init__(self):
            self.picks, self.fail, self.last_pick, self._source = 0, True, {}, None

        def pick_topic(self):
            self.picks += 1
            self.last_pick, self._source = {"source_title": "Abbaye X"}, {"title": "Abbaye X", "extract": "..."}
            return "Titre validé"

        def generate_hook_variants(self, topic, n=5, previous_stats_list=None):
            return [{"text": "Accroche validée", "pattern": "chiffre", "score": 8}]

        def generate_script(self, topic, chosen_hook=None):
            if self.fail:
                raise RuntimeError("quotas épuisés")
            return {"scenes": [], "title": topic}

        def export_state(self):
            return {"source": self._source, "last_pick": self.last_pick}

        def restore_state(self, state):
            self._source, self.last_pick = state["source"], state["last_pick"]

    class FakeChannel:
        def __init__(self, brain):
            self.brain = brain

        def section(self, key):
            return {}

    first = FakeBrain()
    g = gb.GlobalBrain.__new__(gb.GlobalBrain)
    g.channels, g.output_dir = {"mystery": FakeChannel(first)}, tmp_path
    with pytest.raises(RuntimeError):
        g.run_channel("mystery", previous_stats_list=[])
    entry = pending_topic.load_pending("mystery")
    assert entry["topic"] == "Titre validé" and entry["script_attempts"] == 1 and saved == []

    second = FakeBrain()                                     # exécution suivante : nouveau processus
    second.fail = False
    g.channels = {"mystery": FakeChannel(second)}
    path = g.run_channel("mystery", previous_stats_list=[])
    assert second.picks == 0                                 # aucun nouveau choix de sujet
    assert second.last_pick == {"source_title": "Abbaye X"}
    script = json.loads(path.read_text(encoding="utf-8"))
    assert script["hook"]["text"] == "Accroche validée" and script["topic"] == "Titre validé"
    assert pending_topic.load_pending("mystery") is None and len(saved) == 1


def test_pending_topic_is_dropped_after_too_many_failures(monkeypatch, tmp_path):
    from auracontent.utils import pending_topic

    monkeypatch.setattr(pending_topic, "PENDING_PATH", tmp_path / "p.json")
    pending_topic.save_pending("mystery", {"topic": "T"})
    for _ in range(3):
        pending_topic.note_script_attempt("mystery")
    assert pending_topic.load_pending("mystery", max_attempts=3) is None
    assert pending_topic.load_pending("mystery") is None                 # supprimé


def test_failed_run_pushes_only_pending_topic(monkeypatch, tmp_path):
    from auracontent.services.publishing import state_sync

    uploaded = []

    class FakeApi:
        def upload_file(self, **k):
            uploaded.append(k["path_in_repo"])

    monkeypatch.setattr(state_sync, "_api", lambda: (FakeApi(), "t"))
    monkeypatch.setattr(state_sync, "STATE_DIR", tmp_path)
    for name in ("pending_topics.json", "global_history.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    state_sync.push_state(("pending_topics.json",))
    assert uploaded == ["meta/pending_topics.json"]               # jamais les historiques après un échec
    assert "pending_topics.json" in state_sync.STATE_FILES        # récupéré au début de l'exécution suivante


def test_prefilter_skips_llm_when_nothing_usable():
    from auracontent.channels.mystery.brain import prefilter_candidates

    region = {"name": "Bretagne", "search": ["Finistère", "Morbihan"]}
    candidates = [
        {"title": "Quimper", "extract": "Quimper est une commune française du Finistère."},
        {"title": "Trésor de Y", "extract": "Le trésor de Y fut découvert en 1820 près de Vannes, dans le Morbihan."},
        {"title": "Trésor de Z", "extract": "Le trésor de Z fut découvert à Reims."},
    ]
    assert [c["title"] for c in prefilter_candidates(candidates, region)] == ["Trésor de Y"]

    brain = _mystery_brain(_ScriptedLLM())                   # aucune réponse prévue : tout appel échouerait
    import auracontent.channels.mystery.brain as mb
    original = mb.find_candidates
    mb.find_candidates = lambda q, limit, exclude: candidates[:1] + candidates[2:]
    try:
        assert brain._find_source(region, {"name": "c", "search": ["trésor"]}, set()) is None
    finally:
        mb.find_candidates = original
    assert brain.llm.calls == []


LSV_WIKITEXT = """{{Palette Le saviez-vous}}
* <!--@ID_100-->[[Fichier:Phare.jpg|40px|droite|Le phare.]]Le '''[[Phare de la Caravelle|phare de la Caravelle]]''' ''(photo)'', haut de {{Unité|21|m}}, éclaire la rive américaine de l'océan Atlantique.
*:<small>présente 6 jours en page d'accueil. [[Discussion:Phare de la Caravelle/LSV_100|Discussion.]]</small>
* <!--@ID_101-->Le premier monument du {{s|XX}} protégé en France est une plate-forme de canon.
*:<small>[[Discussion:Plate-forme de la pièce d'artillerie 380/LSV_101|Discussion.]]</small>
*:<small>ligne de détail sans anecdote</small>
"""


def test_le_saviez_vous_anecdotes_are_parsed():
    from auracontent.services.scraping.anecdotes import parse_anecdotes

    items = parse_anecdotes(LSV_WIKITEXT)
    assert [(a["id"], a["target"]) for a in items] == [("100", "Phare de la Caravelle"),
                                                       ("101", "Plate-forme de la pièce d'artillerie 380")]
    assert items[0]["text"] == ("Le phare de la Caravelle, haut de 21 m, éclaire la rive américaine "
                                "de l'océan Atlantique.")
    assert "XXe siècle" in items[1]["text"]


@pytest.mark.parametrize("claims, expected", [
    ({"P625": [{"mainsnak": {"datavalue": {"value": {"latitude": 14.8, "longitude": -60.9}}}}]}, True),
    ({"P625": [{"mainsnak": {"datavalue": {"value": {"latitude": -22.9, "longitude": -43.2}}}}],   # au Brésil
      "P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q142"}}}}]}, False),
    ({"P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q142"}}}}],
      "P276": [{"mainsnak": {"datavalue": {"value": {"id": "Q19675"}}}}]}, True),           # objet au Louvre
    ({"P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q142"}}}}]}, False),             # organisation
    ({"P27": [{"mainsnak": {"datavalue": {"value": {"id": "Q142"}}}}]}, False),             # personne
])
def test_anecdote_must_be_a_place_or_object_in_france(claims, expected):
    from auracontent.services.scraping.anecdotes import _in_france
    from auracontent.services.visuals.map_scene import find_region

    assert _in_france(claims, lambda lat, lon: find_region(lat, lon, max_km=5) is not None) is expected


def test_anecdotes_skip_already_used_and_foreign(monkeypatch):
    from auracontent.services.scraping import anecdotes as an

    monkeypatch.setattr(an, "fetch_wikitext", lambda title: LSV_WIKITEXT if "accueil" in title else "")
    monkeypatch.setattr(an, "_wikibase_items", lambda titles: {t: (t, f"Q{i}") for i, t in enumerate(titles)})
    monkeypatch.setattr(an, "located_in_france", lambda qids: set(qids))
    kept = an.collect_french_anecdotes([2020], archives=1, exclude_ids={"101"})
    assert [a["target"] for a in kept] == ["Phare de la Caravelle"]
    kept = an.collect_french_anecdotes([2020], exclude_titles={"phare de la caravelle"})
    assert [a["target"] for a in kept] == ["Plate-forme de la pièce d'artillerie 380"]   # article déjà traité


def test_topic_sources_try_anecdotes_first_then_keywords(monkeypatch):
    import auracontent.channels.mystery.brain as mb

    brain = _mystery_brain()
    brain._anecdotes = [{"id": "7", "text": "Le phare éclaire l'Amérique.", "target": "Phare de la Caravelle"},
                        {"id": "8", "text": "Trop court.", "target": "Article court"}]
    pages = {"Phare de la Caravelle": {"title": "Phare de la Caravelle", "url": "u", "extract": "x",
                                       "coordinates": {"lat": 14.77, "lon": -60.88}, "wikibase_item": "Q1"}}
    monkeypatch.setattr(mb, "fetch_page", lambda title: pages.get(title))
    keyword_calls = []
    brain._find_source = lambda region, category, used: keyword_calls.append(1) or {"title": "Mot-clé"}
    sources = ["le_saviez_vous", "keywords"]
    region, category = {"name": "Bretagne", "search": []}, {"name": "cat", "search": []}

    page, reg, cat = brain._next_source(sources, region, category, set())
    assert page["anecdote"] == "Le phare éclaire l'Amérique." and page["anecdote_id"] == "7"
    assert reg["name"].startswith("Outre-mer") and cat["name"].startswith("anecdote")
    assert keyword_calls == []
    page, reg, cat = brain._next_source(sources, region, category, set())   # anecdote 8 : page inutilisable
    assert page == {"title": "Mot-clé"} and reg == region and keyword_calls == [1]


@pytest.mark.parametrize("lat, lon, expected", [(48.857, 2.352, "Paris intra-muros"), (48.80, 2.13, "Île-de-France"),
                                                (47.24, 6.36, "Bourgogne-Franche-Comté"), (43.30, 5.37, "Provence")])
def test_anecdote_region_matches_yaml_regions(lat, lon, expected):
    brain = _mystery_brain()
    assert brain.region_of({"coordinates": {"lat": lat, "lon": lon}})["name"].startswith(expected)


# ---------- Finitions de la carte (fix/finitions) ----------
@pytest.mark.parametrize("name, lat, lon, landmark, kind", [
    ("Reims", 49.2583, 4.0317, None, None),                   # le sujet est dans la grande ville
    ("Saint-Denis", -20.8789, 55.4481, None, None),           # idem outre-mer
    ("Ajaccio", 41.9192, 8.7386, "BASTIA", "préfecture"),     # aucune grande ville : préfecture la plus proche
    ("Abbaye", 47.2436, 6.3656, "BESANÇON", "ville"),
])
def test_landmark_big_city_then_prefecture(name, lat, lon, landmark, kind):
    from auracontent.services.visuals.map_scene import MapRenderer

    r = MapRenderer(lat, lon, name)
    assert (r.landmark or {}).get("name") == landmark and (r.landmark or {}).get("kind") == kind
    if r.landmark:
        x, y = r.to_screen(r.end, r.landmark["lon"], r.landmark["lat"])
        assert 40 <= x <= 930 and 620 <= y <= 1400


def test_prefectures_cover_departments_and_overseas():
    from auracontent.services.visuals.map_scene import load_prefectures

    prefectures = load_prefectures()
    assert len(prefectures) >= 101 and {"Ajaccio", "Bastia", "Privas", "Foix", "Mamoudzou"} <= {p["name"] for p in prefectures}


@pytest.mark.parametrize("lat, lon", [(-20.8789, 55.4481), (14.6, -61.07), (16.0, -61.73), (-12.78, 45.23)])
def test_overseas_island_fills_a_third_of_the_screen(lat, lon):
    """Réunion, Martinique, Guadeloupe, Mayotte : l'île du sujet occupe au moins un tiers de la largeur."""
    from auracontent.services.visuals.map_scene import MapRenderer

    r = MapRenderer(lat, lon, "Lieu")
    assert r._island_width() * r.end.scale >= 1080 / 3
    x, y = r.to_screen(r.end, lon, lat)
    assert (round(x), round(y)) == r.ANCHOR                     # le point reste au même endroit


# ---------- Mascotte (feat/mascotte) ----------
MASCOT_DIR = "assets/mystery/images"


def test_mascot_is_off_unless_configured():
    from auracontent.channels.base_channel import load_channel_config
    from auracontent.services.visuals.mascot import mascot_files, mascot_settings

    assert mascot_settings(None)["enabled"] is False and mascot_files(mascot_settings({}), "mystery") == {}
    for channel in ("finance", "kids"):                      # autres chaînes : pas de section `mascot`
        assert mascot_settings((load_channel_config(channel) or {}).get("mascot"))["enabled"] is False
    cfg = mascot_settings(load_channel_config("mystery")["mascot"])
    assert cfg["enabled"] and set(mascot_files(cfg, "mystery")) == {"curieux", "surpris", "clin"}


def test_mascot_expression_per_scene():
    from auracontent.services.visuals.mascot import expression_for, mascot_settings

    cfg = mascot_settings({"enabled": True})
    scenes = [{"role": "hook"}, {"role": "context"}, {"role": "reveal"}, {"role": "tension"}, {"role": "cta"}]
    # « surpris » n'est jamais choisi d'après le rôle de la scène : seulement pendant la phrase de la révélation
    assert [expression_for(s, i, len(scenes), cfg, skip={1}) for i, s in enumerate(scenes)] == \
        ["curieux", None, "curieux", "curieux", "clin"]
    assert expression_for({"role": "reveal"}, 4, 5, cfg) == "clin"            # la dernière scène prime


def test_mascot_lands_tip_on_the_place_with_a_bounce():
    from auracontent.services.visuals.mascot import landing, load_mascot, tip_of
    from auracontent.services.visuals.map_scene import MapRenderer

    img = load_mascot(f"{MASCOT_DIR}/mascotte_curieux.png", 190)
    tx, ty = tip_of(img)
    assert img.height == 190 and abs(tx - img.width / 2) < 4 and ty > 0.95 * img.height   # pointe en bas, au centre
    assert landing(0.0, 190)[0] < -100 and landing(1.0, 190)[:3] == (0.0, 1.0, 1.0)   # tombe, puis posée
    contact, mid_hop = landing(0.281, 190), landing(0.42, 190)
    assert contact[0] > -1 and contact[2] < 0.9                           # au contact : écrasée
    assert mid_hop[0] < -25                                               # puis un rebond
    r = MapRenderer(47.6392, 4.3889, "Abbaye de Fontenay", mascot_path=f"{MASCOT_DIR}/mascotte_curieux.png")
    frame = r.render_frame(5.9, 6.0)
    x, y = r.ANCHOR
    rr, g, b = frame.getpixel((x, y - 60))                                # corps de l'épingle au-dessus du lieu
    assert rr > 200 and g > 170 and b < 80
    mx0, my0, mx1, my1 = r._marker_box(x, y)
    lx0, ly0, lx1, ly1 = r._label_box(r.label_below)
    assert ly0 >= my1 or ly1 <= my0                                       # l'étiquette ne couvre pas la mascotte


def test_mascot_corner_clears_headline_subtitles_and_ui():
    from auracontent.channels.base_channel import load_channel_config
    from auracontent.services.video.composer import Composer

    cfg = load_channel_config("mystery")
    c = Composer(cfg["video_composer"], channel_name="mystery", mascot=cfg["mascot"])
    x, y = c.mascot_position()
    lines, _ = c._headline_lines("EN 1793, LA FLÈCHE | DEVAIT ÊTRE RASÉE PAR LA COMMUNE")
    size = c._headline_font_size_for(lines)
    assert len(lines) == 3 and y > c.headline_y + 3 * int(size * 1.34) + int(size * 0.16)
    assert y + cfg["mascot"]["height"] < 1920 - 613 - 110 and x + 100 < 930 and x >= 30


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_scene_shows_the_mascot_in_its_corner(tmp_path):
    import ffmpeg
    from PIL import Image

    from auracontent.services.video.composer import Composer

    image = tmp_path / "bg.jpg"
    Image.new("RGB", (1080, 1920), (20, 30, 60)).save(image)
    voice = tmp_path / "v.wav"
    ffmpeg.input("sine=duration=1", f="lavfi").output(str(voice), ac=2, ar=48000).run(quiet=True)
    c = Composer({"video_width": 1080, "video_height": 1920, "fps": 10}, channel_name="mystery",
                 mascot={"enabled": True})
    c.temp_dir = str(tmp_path)
    out = c.process_scene({"id": "m", "audio_path": str(voice), "duration": 1.0, "mascot": "surpris"}, str(image))
    png = tmp_path / "f.png"
    ffmpeg.input(out, ss=0.5).output(str(png), vframes=1).run(quiet=True)
    x, y = c.mascot_position()
    r, g, b = Image.open(png).convert("RGB").getpixel((x + 30, y + 75))    # corps jaune de l'épingle
    assert r > 200 and g > 160 and b < 90


def test_pipeline_assigns_mascot_and_skips_the_map_scene():
    from auracontent.core import pipeline as pl

    class FakeChannel:
        def __init__(self, mascot):
            self.mascot = mascot

        def section(self, key):
            return {"mascot": self.mascot}.get(key, {})

    vp = pl.VideoPipeline.__new__(pl.VideoPipeline)
    vp.channel = FakeChannel({"enabled": True})
    reveal = "La statue reprend le visage de Vercingétorix."
    scenes = [{"id": 1, "role": "hook"}, {"id": 2, "role": "context", "map": {"lat": 1}},
              {"id": 3, "role": "context", "duration": 8.0, "reveal_sentence": reveal,
               "text": "Le sculpteur part d'un modèle connu. " + reveal},
              {"id": 4, "role": "reveal", "text": "Fin.", "duration": 3}]
    vp._assign_mascot(scenes)
    assert [s.get("mascot") for s in scenes] == ["curieux", None, "curieux", "clin"]
    start, end = scenes[2]["mascot_surprise"]              # « surpris » pendant la phrase de la révélation
    assert 3.5 < start < 4.0 and end == 8.0
    assert "mascot_surprise" not in scenes[3]              # pas d'après le rôle « reveal »
    vp.channel = FakeChannel({})                                          # désactivée : rien n'est posé
    fresh = [{"role": "hook"}]
    vp._assign_mascot(fresh)
    assert "mascot" not in fresh[0]


@pytest.mark.parametrize("title, categories, excluded", [
    ("File:32194-Larressingle-Routes-Hydro.png", "Larressingle|Road maps of communes in Gers", True),
    ("File:Sols.png", "", True),
    ("File:argile.jpg", "", True),
    ("File:Larressingle - Église - 01.jpg", "Église Saint-Sigismond de Larressingle", False),
    ("File:Vue du village.jpg", "Geological maps of Gers", True),          # catégorie seule
    ("File:Église, plan au sol.jpg", "", True),
    ("File:Bas-relief du portail.jpg", "", False),
    ("File:Carte postale ancienne de l'abbaye.jpg", "Cartes postales de la Marne", False),
    ("File:Église 2025.jpg", "Files with coordinates missing SDC location of creation", False),
])
def test_maps_plans_and_diagrams_are_excluded(title, categories, excluded):
    from auracontent.services.visuals.subject_images import is_map_or_diagram

    assert is_map_or_diagram(title, categories) is excluded


def test_png_excluded_when_channel_allows_only_jpeg():
    from auracontent.services.visuals.subject_images import usable

    info = {"mime": "image/png", "width": 3000, "height": 2000,
            "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}}}
    assert usable(info, "File:Vue.png")
    assert not usable(info, "File:Vue.png", allowed_mime={"image/jpeg"})


def test_object_names_match_files_and_titles():
    from auracontent.services.visuals.subject_images import is_distinctive, mentions_focus, name_matches

    place = ["Larressingle", "Gers"]
    assert name_matches("Larressingle (32) Église Saint-Sigismond 11.JPG", "église Saint-Sigismond", place)
    assert not name_matches("Larressingle Gers 6 mairie.jpg", "église Saint-Sigismond", place)
    assert mentions_focus("Larressingle : la statue du saint patron copie un Gaulois",
                          ["statue de saint Sigismond", "église Saint-Sigismond"], place)
    assert not mentions_focus("Larressingle, le village fortifié du Gers", ["statue de saint Sigismond"], place)
    assert is_distinctive("Pont du Gard") and not is_distinctive("Abbaye X")


def test_object_images_come_first_then_the_commune(monkeypatch):
    from auracontent.services.visuals import subject_images as si

    def img(name, origin, desc=""):
        return si.SubjectImage(name, f"File:{name}", f"u/{name}", 2000, 1500, origin, desc or name)

    monkeypatch.setattr(si, "object_images", lambda names, place, files, limit, **k: [img("Statue.jpg", "object")])
    monkeypatch.setattr(si, "article_images", lambda title, **k: [
        img("Larressingle Gers 1.jpg", "article"), img("Larressingle (32) Église Saint-Sigismond 02.JPG", "article")])
    monkeypatch.setattr(si, "category_images", lambda cat, limit, **k: [img("Mairie.jpg", "commons")])
    images = si.collect_subject_images("Larressingle", "Larressingle", {"documents": False},
                                       place_terms=["Larressingle", "Gers"],
                                       focus=["statue de saint Sigismond", "église Saint-Sigismond"])
    assert [i.origin for i in images] == ["object", "article", "object", "commons"]
    assert si.focus_image_count(images) == 2
    pool = si.SubjectImagePool(images)
    assert [i.key for i in pool.candidates()][:2] == ["Statue.jpg", "Larressingle (32) Église Saint-Sigismond 02.JPG"]


def test_topic_without_object_image_is_discarded(monkeypatch):
    from auracontent.channels.mystery import brain as mb

    for name in ("load_global_history",):
        monkeypatch.setattr(mb, name, lambda: [])
    monkeypatch.setattr(mb, "load_used_sources", lambda: set())
    monkeypatch.setattr(mb, "load_rejected", lambda channel: (set(), set()))
    monkeypatch.setattr(mb, "load_last_regions", lambda **k: [])
    monkeypatch.setattr(mb, "load_last_categories", lambda **k: [])
    monkeypatch.setattr(mb, "place_details", lambda qid: {"commune": "C"})
    pages = iter([{"title": f"Lieu {i}", "url": "u", "extract": "x"} for i in range(10)])
    brain = _mystery_brain()
    brain._anecdotes = []
    brain._find_source = lambda region, category, used: next(pages)
    brain.count_source_images = lambda source, details, focus=None: (30, 0)    # 30 photos, aucune de l'objet
    brain._accept_title = lambda trial, used: "Titre"
    with pytest.raises(RuntimeError):
        brain.pick_topic(max_attempts=3)


def test_commune_article_is_its_own_commune(monkeypatch):
    from auracontent.services.scraping import wikipedia as wp

    entities = {
        "Q1": {"labels": {"fr": {"value": "Larressingle"}},
               "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q484170"}}}}],
                          "P131": [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": "Q2"}}}}]}},
        "Q2": {"labels": {"fr": {"value": "Gers"}},
               "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q6465"}}}}]}},
    }
    monkeypatch.setattr(wp, "_wikidata", lambda params: {"entities": {params["ids"]: entities[params["ids"]]}})
    assert wp.place_details("Q1") == {"commune": "Larressingle", "department": "Gers"}   # pas « canton de... »


def test_anecdote_keeps_its_illustration_and_cited_articles():
    from auracontent.services.scraping.anecdotes import parse_anecdotes

    line = ("* <!--@ID_14652-->[[Fichier:Larressingle (32) Église Saint-Sigismond 11.JPG|40px|droite|Statue de "
            "[[Sigismond (saint)|saint Sigismond]] dans l’église de [[Larressingle]].]]Dans l’église Saint-Sigismond "
            "de '''[[Larressingle]]''', la statue du [[Sigismond (saint)|saint patron]] ''(photo)'' est une réduction "
            "de [[Vercingétorix (Millet)|celle de Vercingétorix]].")
    a = parse_anecdotes(line)[0]
    assert a["target"] == "Larressingle" and a["file"] == "Larressingle (32) Église Saint-Sigismond 11.JPG"
    assert a["caption"] == "Statue de saint Sigismond dans l'église de Larressingle."
    assert a["links"] == ["Sigismond (saint)", "Vercingétorix (Millet)"]


def test_corner_mascot_has_no_drop_shadow():
    from auracontent.services.visuals.mascot import load_mascot

    path = "assets/mystery/images/mascotte_curieux.png"
    with_shadow, without = load_mascot(path, None, True), load_mascot(path, None, False)

    def faint(img):
        a = img.getchannel("A")
        return sum(1 for y in range(int(img.height * 0.88), img.height) for x in range(img.width)
                   if 0 < a.getpixel((x, y)) <= 70)

    assert faint(with_shadow) > 2000 and faint(without) < 300
    assert without.height <= with_shadow.height


def test_documents_get_reserved_shots(monkeypatch, tmp_path):
    m = _fake_manager(monkeypatch, tmp_path, [("a1", "article"), ("a2", "article"), ("a3", "article"),
                                              ("d1", "document")])
    m.subject_cfg["document_shots"] = 1
    out = str(tmp_path / "x")
    origins = []
    for sid in (1, 2, 3, 4):
        m.get_scene_asset({"id": sid}, out)
        origins.append(m.last_origin)
    assert origins == ["article", "article", "document", "article"]     # même si les photos suffisent


def test_archive_queries_widen_to_the_building_name(monkeypatch, tmp_path):
    m = _fake_manager(monkeypatch, tmp_path, [])
    m.subject.update(focus=["statue de saint Sigismond", "église Saint-Sigismond"], commune="Larressingle",
                     place_terms=["Larressingle", "Gers"])
    assert m.archive_queries({}) == ["église Saint-Sigismond Larressingle", "église Saint-Sigismond"]
    m.subject.update(focus=["Abbaye de la Grâce-Dieu"], place_terms=["Chaux-lès-Passavant", "Doubs"])
    assert m.archive_notice_ok("Gravure de l'abbaye de la Grâce-Dieu", "Abbaye de la Grâce-Dieu")   # distinctif
    assert not m.archive_notice_ok("Abbaye X, Gironde", "Abbaye X")
