"""Cerveau de la chaîne Mystery.

Flux : région + catégorie + format imposés par le code -> page Wikipédia réelle
-> titre écrit à partir de la page -> hooks -> script ancré sur la page -> fact-check.
"""
import difflib
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from auracontent.core.brain.base_brain import BaseBrain, SceneCountError, VeracityIssue
from auracontent.services.scraping.anecdotes import collect_french_anecdotes
from auracontent.services.scraping.wikipedia import fetch_intro, fetch_page, find_candidates, place_details
from auracontent.services.visuals.asset_manager import subject_settings
from auracontent.services.visuals.subject_images import (
    WEAK_NAME_WORDS,
    collect_subject_images,
    focus_image_count,
    mentions_focus,
    name_matches,
    name_tokens,
    place_terms_of,
)
from auracontent.utils.pending_topic import load_rejected
from auracontent.utils.topic_tracker import (
    load_global_history,
    load_last_categories,
    load_last_regions,
    load_used_sources,
    load_used_values,
)

MOODS = {"ominous", "intriguing", "tense", "awe", "scientific", "investigation"}
ROLES = {"hook", "tension", "context", "value", "escalation", "reveal", "cta"}
SCENE_TYPES = {"specific", "generic"}
# (format, condition sur la source) : on ne propose pas « commence par l'année » sans date dans la source
TITLE_FORMATS = [
    ("fait daté (commence par l'année ou la date)", "date"),
    ("lieu en tête, suivi de deux-points puis du fait", None),
    ("paradoxe en une phrase affirmative", None),
    ("chiffre précis en tête", "number"),
]
MAX_TITLE_CHARS = 80              # au-delà, YouTube coupe le titre bien avant la fin
BANNED_TITLE_WORDS = ("le plus ", "la plus ", "les plus ", "incroyable", "hallucinant", "bizarre")
MAX_WORDS_PER_SCENE = 32          # le prompt demande deux phrases de 12 mots au plus, on tolère jusqu'à 32
DEFAULT_TOPIC_ATTEMPTS = 10
ANECDOTE_CATEGORY = {"name": "anecdote « Le saviez-vous ? » de Wikipédia", "search": []}
DEFAULT_ANECDOTE_YEARS = (2010, 2025)
DEFAULT_MIN_SOURCE_IMAGES = 5
HOOK_ROUNDS = 3                   # générations d'accroches avant d'abandonner (jamais d'accroche par défaut)
MAX_HOOK_WORDS = 12               # l'accroche ouvre la scène 1 : même limite que les phrases du script
NUMBER_WORDS = {"deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze",
                "treize", "quatorze", "quinze", "seize", "vingt", "trente", "quarante", "cinquante",
                "soixante", "cent", "cents", "mille", "million", "millions", "milliard", "milliards"}
STOPWORDS = {"le", "la", "les", "l", "de", "du", "des", "d", "et", "en", "au", "aux", "a", "à", "un", "une",
             "sur", "sous", "dans", "par", "pour", "ce", "cet", "cette", "son", "sa", "ses", "sans"}
NUMBER_VALUES = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9,
                 "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
                 "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60, "cent": 100,
                 "cents": 100, "mille": 1000, "million": 1000000, "millions": 1000000,
                 "milliard": 1000000000, "milliards": 1000000000}
DIGITS_RE = re.compile(r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?!\d)|\d+")
ROMAN_RE = re.compile(r"\b([IVXL]{1,6})(?:e|er|ème|eme)\b")
YEAR_RE = re.compile(r"(?<!\d)(?:[3-9]\d{2}|1\d{3}|20[0-2]\d)(?!\d)")
MONTHS = r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
DATE_RE = re.compile(rf"\b\d{{1,2}}(?:er)?\s+(?:{MONTHS})\b|\b(?:{MONTHS})\s+\d{{3,4}}\b", re.IGNORECASE)
DEFAULT_MIN_FACTS = 4
DEFAULT_MIN_SCENES = 3               # brain.min_scenes : plancher du script (cible et résultat)


def fold(text: str) -> str:
    """Minuscules, sans accents ni ponctuation : comparaisons robustes (titres, accroches, lieux)."""
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


_SPECIAL_CHARS = {"\u202f": " ", "\u00a0": " ", "\u2011": "-", "\u2010": "-"}
REVEAL_MIN_SIMILARITY = 0.3          # en dessous : dernière phrase de la scène « reveal »


def title_key(text: str) -> str:
    """Forme de comparaison des titres (comme finance) : NFKC, espaces fines/insécables et tirets insécables
    remplacés, contenu entre parenthèses ignoré, sans accents, casse ni ponctuation.
    « Poissons (Haute-Marne) » -> « poissons »."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    for char, repl in _SPECIAL_CHARS.items():
        text = text.replace(char, repl)
    return fold(re.sub(r"\([^)]*\)", " ", text))


def title_words(title: str) -> int:
    """Mots d'un titre après normalisation (NFKC, espaces fines/insécables) : la ponctuation seule (« : »,
    « ? ») ne compte pas, et un nombre écrit avec séparateur de milliers (« 4 101 ») compte pour un mot."""
    text = unicodedata.normalize("NFKC", str(title or ""))
    for char, repl in _SPECIAL_CHARS.items():
        text = text.replace(char, repl)
    text = re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)
    return len([w for w in text.split() if re.search(r"\w", w)])


class TitleSet(set):
    """Ensemble de titres comparés par title_key : « Boussac (Creuse) » et « boussac » sont le même titre."""

    def __init__(self, items=()):
        super().__init__(title_key(t) for t in items)

    def add(self, item) -> None:
        super().add(title_key(item))

    def __contains__(self, item) -> bool:
        return super().__contains__(title_key(item))

    def __or__(self, other) -> "TitleSet":
        return TitleSet([*self, *other])


def similarity(a: str, b: str) -> float:
    """Proximité de deux phrases après normalisation : le meilleur du recouvrement des mots (F1) et du
    rapport de séquence (difflib)."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return 0.0
    wa, wb = {w for w in fa.split() if len(w) > 2}, {w for w in fb.split() if len(w) > 2}
    common = len(wa & wb)
    f1 = 2 * common / (len(wa) + len(wb)) if wa and wb else 0.0
    return max(f1, difflib.SequenceMatcher(None, fa, fb).ratio())


def sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+|[;:]\s+", str(text)) if p.strip()]


def count_words(sentence: str) -> int:
    """Mots prononcés : « l'abbaye » compte pour un mot, la ponctuation seule pour aucun."""
    return len([w for w in re.split(r"\s+", sentence) if re.search(r"\w", w)])


# Articles généralistes : la première phrase définit une entité administrative, pas un fait
GENERALIST = re.compile(r"\best (une|un|l une des|l un des) (commune|ville|region|province|departement|localite|"
                        r"agglomeration|metropole|collectivite|canton|arrondissement|pays|intercommunalite)\b")


def prefilter_candidates(candidates: list[dict], region: dict) -> list[dict]:
    """Filtre sans modèle avant le choix par LLM : écarte les articles généralistes (commune, département...)
    et les pages qui ne nomment ni la région visée ni l'un de ses lieux de recherche."""
    places = [fold(x) for x in [region.get("name", "").split("(")[0], *region.get("search", [])] if fold(x)]
    kept = []
    for c in candidates:
        head = fold(c.get("extract", "")[:300])
        if GENERALIST.search(head):
            continue
        text = f" {fold(c.get('title', ''))} {fold(c.get('extract', '')[:4000])} "
        if places and not any(f" {p} " in text for p in places):
            continue
        kept.append(c)
    return kept


def banned_word_pattern(words) -> re.Pattern | None:
    """« escarpé » attrape aussi « escarpée », « escarpés », « escarpées »."""
    stems = [re.escape(w.strip().lower()) for w in words or [] if str(w).strip()]
    if not stems:
        return None
    return re.compile(r"\b(" + "|".join(stems) + r")(e|s|es|x)?\b", re.IGNORECASE)


def _roman(text: str) -> int | None:
    values = {"I": 1, "V": 5, "X": 10, "L": 50}
    total = 0
    for i, ch in enumerate(text):
        v = values[ch]
        total += -v if i + 1 < len(text) and values[text[i + 1]] > v else v
    return total or None


def numbers_in(text: str) -> set[str]:
    """Chiffres et dates écrits en chiffres (« 1 200 » -> « 1200 ») ou en siècle romain (« XIXe » -> « 19 »)."""
    found = {str(int(re.sub(r"\D", "", m.group()))) for m in DIGITS_RE.finditer(str(text))}
    found |= {str(v) for m in ROMAN_RE.finditer(str(text)) if (v := _roman(m.group(1)))}
    return found


def source_has_date(text: str) -> bool:
    return bool(YEAR_RE.search(str(text)) or DATE_RE.search(str(text)))


def proper_nouns(text: str) -> set[str]:
    """Noms propres de la source, sans accents : mots à majuscule qui n'ouvrent pas une phrase."""
    found = set()
    for m in re.finditer(r"[^\W\d_][\w'\u2019-]*", str(text)):
        word = m.group()
        if not word[:1].isupper():
            continue
        before = str(text)[:m.start()].rstrip()
        if not before or before[-1] in ".!?:«\"\n":
            continue                                   # première lettre de phrase : pas forcément un nom
        found.update(t for t in fold(word).split() if len(t) >= 3 and t not in STOPWORDS
                     and t not in WEAK_NAME_WORDS and not ROMAN_RE.fullmatch(t.upper()))
    return found


def claim_issues(text: str, source: str, what: str = "le texte") -> list[str]:
    """Titre, accroche, bandeau : une date ou un chiffre n'est permis que s'il figure mot pour mot dans la
    source, et le texte contient un chiffre, une date ou un nom propre présent dans la source."""
    issues = []
    src_numbers, src_words = numbers_in(source), set(fold(source).split())
    numbers = numbers_in(text)
    missing = sorted(n for n in numbers if n not in src_numbers)
    words = fold(text).split()
    missing += [w for w in words if w in NUMBER_VALUES and w not in src_words
                and str(NUMBER_VALUES[w]) not in src_numbers]
    if missing:
        issues.append(f"{what} « {text} » contient {', '.join(missing)}, absent(s) de la source : une date ou un "
                      "chiffre n'est permis que s'il figure mot pour mot dans la source.")
    has_number = bool(numbers) or any(w in NUMBER_VALUES for w in words)
    if not has_number and not (set(words) & proper_nouns(source)):
        issues.append(f"{what} « {text} » ne contient ni chiffre, ni date, ni nom propre présent dans la source.")
    return issues


def title_format(source_text: str, rng=random) -> str:
    """Format de titre tiré au hasard parmi ceux que la source permet (date, chiffre)."""
    allowed = [label for label, need in TITLE_FORMATS
               if need is None or (need == "date" and source_has_date(source_text))
               or (need == "number" and numbers_in(source_text))]
    return rng.choice(allowed)


def source_text(source: dict | None, max_chars: int = 6000) -> str:
    """Source transmise au script et au contrôle de véracité : anecdote « Le saviez-vous ? » (texte et
    légende de son illustration), article qu'elle cite en gras, puis les autres articles qu'elle cite."""
    if not source:
        return ""
    parts = []
    if source.get("anecdote"):
        parts.append(f"ANECDOTE « Le saviez-vous ? » de Wikipédia : « {source['anecdote']} »")
        if source.get("anecdote_caption"):
            parts.append(f"LÉGENDE DE SON ILLUSTRATION : « {source['anecdote_caption']} »")
    parts.append(f"ARTICLE « {source.get('title', '')} » :\n{str(source.get('extract', ''))[:max_chars]}")
    for rel in source.get("related") or []:
        parts.append(f"ARTICLE CITÉ PAR L'ANECDOTE « {rel['title']} » :\n{rel['extract']}")
    return "\n\n".join(parts)


@dataclass
class MysteryBrain(BaseBrain):
    channel_name: str = "mystery"
    _source: dict | None = field(default=None, init=False, repr=False)      # page Wikipédia du sujet
    _anecdotes: list | None = field(default=None, init=False, repr=False)   # réserve « Le saviez-vous ? »
    last_pick: dict = field(default_factory=dict, init=False, repr=False)   # région, catégorie, source
    _report: Counter = field(default_factory=Counter, init=False, repr=False)  # sujets testés, rejets par motif

    @property
    def prompts_dir(self) -> Path:
        return Path(__file__).parent / "prompts"

    # ==========================================
    # SUJET
    # ==========================================
    def build_title_vars(self, used_topics, stats_text=""):
        b = self.config["brain"]

        # régions et catégories : {name, search} dans le YAML ; on compare sur 'name'
        last_regions = load_last_regions(n=b["region_lookback"], channel=self.channel_name)
        region = random.choice([r for r in b["regions"] if r["name"] not in last_regions] or b["regions"])

        last_cats = load_last_categories(n=2, channel=self.channel_name)
        category = random.choice(
            [c for c in b["theme_categories"] if c["name"] not in last_cats] or b["theme_categories"]
        )

        recent = " ".join(used_topics[-8:]).lower()
        ban_hits = sum(recent.count(w.lower()) for w in b["banned_if_recent"])
        ban_block = ""
        if ban_hits >= 2:
            ban_block = ("INTERDICTION TEMPORAIRE STRICTE : ces concepts ont été trop utilisés : "
                         + ", ".join(b["banned_if_recent"]) + ". N'utilise AUCUN de ces mots ni concepts.")

        used_block = ""
        if used_topics:
            used_block = ("SUJETS DÉJÀ TRAITÉS (propose un angle, un vocabulaire et un format "
                          "radicalement différents) :\n- " + "\n- ".join(used_topics[-30:]))

        return {
            "forced_region": region["name"],
            "forced_category": category["name"],
            "forced_format": random.choice(TITLE_FORMATS),   # tiré par le code : le LLM ne peut pas "alterner"
            "ban_block": ban_block,
            "stats_block": stats_text,
            "used_topics_block": used_block,
        }, region, category, ban_hits

    def pick_topic(self, max_attempts: int | None = None, previous_stats_list=None) -> str:
        """Retourne le titre. La page source et la région sont gardées dans self._source / self.last_pick.

        Chaque essai : page Wikipédia -> nombre d'images exploitables du lieu (seuil brain.min_source_images)
        -> titre -> retournement validé par le LLM -> anti-doublon. Si aucun essai ne passe, on retient le
        sujet testé qui a le plus d'images plutôt que d'échouer.
        """
        b = self.config["brain"]
        max_attempts = int(max_attempts or b.get("topic_attempts", DEFAULT_TOPIC_ATTEMPTS))
        min_images = int(b.get("min_source_images", DEFAULT_MIN_SOURCE_IMAGES))
        used = load_global_history()
        rejected_titles, self._rejected_ids = load_rejected(self.channel_name)
        used_sources = TitleSet(load_used_sources() | rejected_titles)   # sujets publiés ou abandonnés
        stats_text = self._stats_text(previous_stats_list)
        tested: dict[str, dict] = {}          # titre de la page -> essai (pour le repli)

        sources = [str(x) for x in b.get("topic_sources") or ["keywords"]]
        for attempt in range(1, max_attempts + 1):
            title_vars, region, category, _ = self.build_title_vars(used, stats_text)
            source, region, category = self._next_source(sources, region, category, used_sources | tested.keys())
            if source is None:
                self._report["rejet : pas de page exploitable"] += 1
                print(f"   ↻ Essai {attempt}/{max_attempts} : aucune page Wikipédia exploitable "
                      f"({category['name']} / {region['name']}).")
                continue
            title_vars.update(forced_region=region["name"], forced_category=category["name"])

            title_vars["forced_format"] = title_format(source_text(source))
            trial = {"source": source, "region": region, "category": category, "title_vars": title_vars}
            trial["details"] = place_details(source.get("wikibase_item"))
            trial["focus"] = self.subject_focus(source)
            trial["images"], trial["focus_images"] = self.count_source_images(source, trial["details"],
                                                                              trial["focus"])
            tested[title_key(source["title"])] = trial
            self._report["sujets testés"] += 1
            print(f"   🖼️ Essai {attempt}/{max_attempts} « {source['title']} » : {trial['images']} image(s) "
                  f"exploitable(s) (minimum {min_images}), dont {trial['focus_images']} de "
                  f"« {' / '.join(trial['focus'])} »")
            if not trial["focus_images"]:
                print("   ↻ Aucune image de l'objet ou de son édifice : sujet écarté.")
                self._report["rejet : images"] += 1
                continue
            if trial["images"] < min_images:
                print("   ↻ Sujet trop peu illustré, sujet suivant.")
                self._report["rejet : images"] += 1
                continue

            title = self._accept_title(trial, used)
            if title:
                return self._retain(trial, title)

        if tested:
            print(f"⚠️ Aucun sujet ne passe tous les contrôles en {max_attempts} essais : "
                  "repli sur le sujet testé le mieux illustré, sans repasser le contrôle du retournement.")
            for trial in sorted((t for t in tested.values() if t["focus_images"]), key=lambda t: -t["images"])[:3]:
                print(f"   ↪ « {trial['source']['title']} » ({trial['images']} image(s))")
                self._report["repli"] += 1
                title = self._accept_title(trial, used, check_reversal=False)
                if title:
                    return self._retain(trial, title)
        raise RuntimeError(f"Aucun sujet valide trouvé après {max_attempts} tentatives.")

    # ==========================================
    # REPRISE (sujet validé, script en échec)
    # ==========================================
    def export_state(self) -> dict:
        return {"source": self._source, "last_pick": dict(self.last_pick)}   # faits et réponse compris

    def restore_state(self, state: dict) -> None:
        self._source = state.get("source")
        self.last_pick = dict(state.get("last_pick") or {})

    def count_source_images(self, source: dict, details: dict, focus: list[str] | None = None) -> tuple[int, int]:
        """(images exploitables, dont images de l'objet ou de l'édifice) : objet de l'anecdote, article
        Wikipédia source et sa catégorie Commons (licence libre, 1000 px de large au moins, ni logo, ni carte,
        ni plan, ni schéma, ni blason). (0, 0) si Wikimedia échoue.

        Sans objet distinct de l'article (sujet trouvé par mots-clés, anecdote sur le monument lui-même),
        toutes les images de l'article et de sa catégorie montrent le sujet."""
        cfg = subject_settings(self.config.get("visuals"))
        distinct = self.focus_is_distinct(source, focus)
        try:
            images = collect_subject_images(source["title"], details.get("commons_category"), cfg,
                                            place_terms=place_terms_of(details), documents=False,
                                            focus=focus if distinct else None,
                                            focus_files=[source["anecdote_file"]] if distinct
                                            and source.get("anecdote_file") else None)
        except Exception as e:   # jamais bloquant : le sujet est simplement considéré comme non illustré
            print(f"⚠️ Comptage des images impossible : {e}")
            return 0, 0
        return len(images), (focus_image_count(images) if distinct else len(images))

    @staticmethod
    def focus_is_distinct(source: dict, focus: list[str] | None) -> bool:
        """L'objet de l'anecdote est-il autre chose que l'article (une statue dans l'église d'une commune) ?"""
        return bool(focus) and not any(name_matches(source.get("title", ""), n) for n in focus)

    def subject_focus(self, source: dict) -> list[str]:
        """Noms de l'objet précis et de son édifice. Anecdote : demandés au LLM, puis vérifiés (chaque nom doit
        se lire dans l'anecdote, sa légende ou l'article). Sinon, ou en cas d'échec : le titre de l'article."""
        default = [re.sub(r"\s*\([^)]*\)\s*$", "", source.get("title", "")).strip()]
        if not source.get("anecdote"):
            return default
        sections = self.read_sections("anecdote_focus.md")
        user = self.fill(sections["user"], anecdote=source["anecdote"], target=source.get("title", ""),
                         caption=source.get("anecdote_caption") or "aucune")
        try:
            with self.llm_stage("objet"):
                data = self.llm.call_json_with_retry(
                    [{"role": "system", "content": sections["system"]}, {"role": "user", "content": user}],
                    temperature=0, max_completion_tokens=200, **self.simple_routing())
        except (RuntimeError, ValueError) as e:
            print(f"⚠️ Objet de l'anecdote non déterminé ({e}) : l'article sert de sujet.")
            return default
        reference = " ".join([source["anecdote"], source.get("anecdote_caption") or "", source.get("title", ""),
                              str(source.get("extract", ""))])
        names = [str(data.get(k) or "").strip() for k in ("objet", "edifice")]
        names = [n for n in names if n and name_matches(reference, n)]
        return names or default

    def _accept_title(self, trial: dict, used, check_reversal: bool = True) -> str | None:
        """Titre écrit d'après la source, avec retournement et sans doublon ; None sinon.
        `check_reversal=False` (repli) : le contrôle du retournement n'est pas repassé, les autres le sont."""
        title = self._write_title(trial["title_vars"], trial["source"], trial.get("focus"))
        if not title:
            self._report["rejet : titre"] += 1
            return None
        issues = claim_issues(title, source_text(trial["source"], max_chars=100000), "titre")
        if issues:
            print(f"   ↻ Titre rejeté ({issues[0]}) : « {title} »")
            self._report["rejet : titre"] += 1
            return None
        focus = trial.get("focus") or []
        if focus and not self.names_subject(title, trial):
            print(f"   ↻ Titre rejeté : « {title} » ne nomme ni l'article « {trial['source'].get('title', '')} » "
                  f"ni l'objet « {' / '.join(focus)} ».")
            self._report["rejet : titre"] += 1
            return None
        ok, reason = self.has_reversal(title, trial["source"]) if check_reversal else (True, "")
        if not ok:
            print(f"   ↻ Titre sans retournement rejeté : « {title} » ({reason or 'motif non fourni par le juge'})")
            self._report["rejet : retournement"] += 1
            return None
        is_dup, matched = self.is_duplicate_topic_llm(title, used)
        if is_dup:
            print(f"   ↻ Doublon avec « {matched} », nouvel essai.")
            self._report["rejet : doublon"] += 1
            return None
        if not self.extract_facts(trial, title):
            self._report["rejet : faits"] += 1
            return None
        return title

    @staticmethod
    def names_subject(title: str, trial: dict) -> bool:
        """Le titre nomme-t-il le sujet ? Le nom de l'article (sans parenthèses : « Poissons (Haute-Marne) »
        -> « Poissons »), OU le nom court de l'objet : un de ses noms propres distinctifs, même si c'est la
        commune (« Stenay » suffit pour « blason de la ville de Stenay »), OU son nom commun (« statue »)."""
        words = f" {title_key(title)} "
        article = title_key(trial.get("source", {}).get("title", ""))
        if article and f" {article} " in words:
            return True
        for name in trial.get("focus") or []:
            proper, _ = name_tokens(name)
            if any(t not in WEAK_NAME_WORDS and len(t) >= 4 and f" {t} " in words for t in proper):
                return True
        return mentions_focus(title, trial.get("focus") or [], place_terms_of(trial.get("details")))

    def extract_facts(self, trial: dict, title: str) -> bool:
        """Faits utilisables sur le sujet précis, tirés de la source (gros modèle : véracité), et réponse à
        la question du titre. Moins de brain.min_facts faits, ou pas de réponse : sujet écarté."""
        b = self.config.get("brain", {})
        # au moins un fait par scène : sans assez de faits pour le plancher de scènes, le sujet est écarté
        min_facts = max(int(b.get("min_facts", DEFAULT_MIN_FACTS)), self.min_scenes())
        sections = self.read_sections("facts.md")
        user = self.fill(sections["user"], title=title, focus=" / ".join(trial.get("focus") or []) or title,
                         source_block=source_text(trial["source"]))
        try:
            with self.llm_stage("faits"):
                data = self.llm.call_json_with_retry(
                    [{"role": "system", "content": sections["system"]}, {"role": "user", "content": user}],
                    temperature=0, max_completion_tokens=1200, **self.script_routing())
        except (RuntimeError, ValueError) as e:
            print(f"   ↻ Extraction des faits impossible ({e}) : sujet écarté.")
            return False
        facts = [str(f).strip() for f in data.get("faits") or [] if str(f).strip()]
        answer = str(data.get("reponse") or "").strip()
        if answer.lower() in ("null", "none", "aucune"):
            answer = ""
        print(f"   📋 {len(facts)} fait(s) utilisable(s) sur le sujet ; réponse au titre : {answer or 'AUCUNE'}")
        if len(facts) < min_facts:
            print(f"   ↻ Moins de {min_facts} faits dans la source (il en faut un par scène, "
                  f"{self.min_scenes()} scènes au moins) : sujet écarté avant le script.")
            return False
        if b.get("require_title_answer", True) and not answer:
            print(f"   ↻ La source ne répond pas à la question du titre ({data.get('question') or '?'}) : sujet écarté.")
            return False
        trial["facts"], trial["answer"] = facts, answer
        return True

    def _retain(self, trial: dict, title: str) -> str:
        source, region, category = trial["source"], trial["region"], trial["category"]
        source = {**source, "facts": trial.get("facts") or [], "answer": trial.get("answer") or "",
                  "focus": trial.get("focus") or []}
        self._source = source
        self.last_pick = {"region": region["name"], "category": category["name"],
                          "source_title": source["title"], "source_url": source["url"],
                          "source_images": trial["images"],
                          "topic_source": "le_saviez_vous" if source.get("anecdote") else "keywords"}
        if source.get("anecdote"):
            self.last_pick.update(anecdote=source["anecdote"], anecdote_id=source.get("anecdote_id"))
        # objet précis et son édifice (images en tête), si ce n'est pas l'article entier
        if self.focus_is_distinct(source, source["focus"]):
            self.last_pick.update(focus=source["focus"], focus_images=trial.get("focus_images"),
                                  focus_files=[source["anecdote_file"]] if source.get("anecdote_file") else [])
        self.last_pick["facts_count"] = len(source["facts"])
        if source.get("coordinates"):   # pour la carte animée (lot 1)
            self.last_pick.update(lat=source["coordinates"]["lat"], lon=source["coordinates"]["lon"])
        # commune (étiquette de la carte, homonymes), département, catégorie Commons (photos du lieu)
        self.last_pick.update(trial.get("details") or {})
        return title

    def has_reversal(self, title: str, source: dict) -> tuple[bool, str]:
        """Le LLM valide que le titre contient un retournement (ce qui s'est passé contredit ce qu'on
        attendrait). Une simple fondation, naissance ou construction est rejetée. En cas d'erreur du LLM,
        le titre est rejeté (jamais accepté par défaut)."""
        sections = self.read_sections("title_check.md")
        user = self.fill(sections["user"], title=title, source_title=source.get("title", ""),
                         source_extract=str(source.get("extract", ""))[:1500])
        try:
            with self.llm_stage("retournement"):
                data = self.llm.call_json_with_retry(
                    [{"role": "system", "content": sections["system"]}, {"role": "user", "content": user}],
                    temperature=0, max_completion_tokens=200, **self.simple_routing(),
                )
        except (RuntimeError, ValueError) as e:
            return False, f"contrôle impossible : {e}"
        return data.get("retournement") is True, str(data.get("raison", "")).strip()

    # ==========================================
    # SOURCES DE SUJETS (brain.topic_sources)
    # ==========================================
    def _next_source(self, sources, region, category, excluded):
        """(page, région, catégorie) : chaque source de brain.topic_sources dans l'ordre, la suivante si
        elle ne donne rien. « le_saviez_vous » : anecdotes de Wikipédia ; « keywords » : recherche aléatoire."""
        for name in sources:
            if name == "le_saviez_vous":
                page = self._next_anecdote(excluded)
                if page:
                    return page, self.region_of(page), ANECDOTE_CATEGORY
            elif name == "keywords":
                page = self._find_source(region, category, excluded)
                if page:
                    return page, region, category
                return None, region, category
            else:
                print(f"⚠️ Source de sujets inconnue dans brain.topic_sources : {name}")
        return None, region, category

    def _next_anecdote(self, excluded) -> dict | None:
        """Prochaine anecdote inédite dont l'article est exploitable ; aucun appel LLM."""
        cfg = dict(self.config.get("brain", {}).get("le_saviez_vous") or {})
        if self._anecdotes is None:
            first, last = cfg.get("years", DEFAULT_ANECDOTE_YEARS)
            try:
                self._anecdotes = collect_french_anecdotes(
                    list(range(int(first), int(last) + 1)), archives=int(cfg.get("archives_per_run", 2)),
                    include_home=bool(cfg.get("include_home", True)), sample=int(cfg.get("sample", 200)),
                    exclude_ids=load_used_values("anecdote_id"), exclude_titles=set(excluded))
            except Exception as e:      # jamais bloquant : la recherche par mots-clés prend le relais
                print(f"⚠️ « Le saviez-vous ? » indisponible : {e}")
                self._anecdotes = []
        excluded = excluded if isinstance(excluded, TitleSet) else TitleSet(excluded)
        while self._anecdotes:
            anecdote = self._anecdotes.pop(0)
            if anecdote["target"] in excluded or str(anecdote.get("id")) in getattr(self, "_rejected_ids", ()):
                continue
            page = fetch_page(anecdote["target"])
            if not page:
                print(f"   ↻ Anecdote écartée (article trop court ou trop général) : {anecdote['target']}")
                continue
            print(f"   💡 Anecdote : {anecdote['text'][:140]}")
            related = []
            for link in (anecdote.get("links") or [])[:int(cfg.get("cited_articles", 3))]:
                intro = fetch_intro(link, int(cfg.get("cited_article_chars", 1500)))
                if intro and title_key(intro["title"]) != title_key(page["title"]):
                    related.append(intro)
            return {**page, "anecdote": anecdote["text"], "anecdote_id": anecdote["id"],
                    "anecdote_file": anecdote.get("file"), "anecdote_caption": anecdote.get("caption") or "",
                    "related": related,
                    "wikibase_item": page.get("wikibase_item") or anecdote.get("wikibase_item"),
                    # coordonnées Wikidata si la page n'en a pas (région, carte animée)
                    "coordinates": page.get("coordinates") or anecdote.get("coordinates")}
        return None

    def region_of(self, page: dict) -> dict:
        """Région du YAML qui contient le lieu (coordonnées de la page), pour la rotation des régions."""
        from auracontent.services.visuals.map_scene import find_region, is_metropolitan

        regions = self.config["brain"]["regions"]
        coords = page.get("coordinates") or {}
        if not coords:
            return {"name": "France", "search": []}
        lat, lon = coords["lat"], coords["lon"]
        if not is_metropolitan(lat, lon):
            return next((r for r in regions if "outre" in fold(r["name"])), {"name": "Outre-mer", "search": []})
        feature = find_region(lat, lon)
        name = fold((feature or {}).get("properties", {}).get("name", ""))
        if name == "ile de france" and 48.815 <= lat <= 48.902 and 2.224 <= lon <= 2.470:
            return next((r for r in regions if fold(r["name"]).startswith("paris")), {"name": "Paris", "search": []})
        for r in regions:
            if name and fold(r["name"]).startswith(name):
                return r
        return {"name": (feature or {}).get("properties", {}).get("name", "France"), "search": []}

    def _find_source(self, region, category, used_sources):
        query = f'{random.choice(category["search"])} {random.choice(region["search"])}'
        print(f"   🔎 Wikipédia : « {query} »")
        candidates = find_candidates(query, limit=8, exclude=used_sources)
        kept = prefilter_candidates(candidates, region)
        if len(kept) < len(candidates):
            print(f"   🧹 Filtre sans modèle : {len(candidates) - len(kept)} page(s) écartée(s) sur {len(candidates)}.")
        if not kept:
            return None                       # rien d'utilisable : aucun appel LLM
        return self._choose_source(kept, region, category)

    def _choose_source(self, candidates, region, category):
        """Le LLM retient la page qui raconte un fait précis de la catégorie, dans la région (ou aucune)."""
        listing = "\n".join(f'{i}. {c["title"]} : {c["extract"][:400]}' for i, c in enumerate(candidates))
        prompt = (
            f"CATÉGORIE : {category['name']}\nRÉGION : {region['name']}\n\nPAGES WIKIPÉDIA :\n{listing}\n\n"
            "Choisis la page qui raconte un fait, un événement, un objet ou une personne PRÉCIS appartenant "
            "à cette catégorie et situé dans cette région, avec assez de matière pour une vidéo d'une minute. "
            "Exclus les articles généralistes (ville, département, région, biographie de personnalité connue). "
            'Si aucune ne convient, réponds -1. Réponds uniquement en JSON : {"index": <entier>}'
        )
        try:
            with self.llm_stage("page"):
                data = self.llm.call_json_with_retry([{"role": "user", "content": prompt}],
                                                     temperature=0, max_completion_tokens=100, **self.simple_routing())
            idx = int(data.get("index", -1))
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"⚠️ Choix de la source impossible : {e}")
            return None
        return candidates[idx] if 0 <= idx < len(candidates) else None

    def _write_title(self, title_vars, source, focus=None):
        sections = self.read_sections("titles.md")
        source_block = f'SOURCE :\n"""{source_text(source, max_chars=3000)}"""'
        if source.get("anecdote"):   # fait déjà repéré et sourcé par les contributeurs de Wikipédia
            source_block = (f"ANGLE À RETENIR (rubrique « Le saviez-vous ? » de Wikipédia) : « {source['anecdote']} »\n"
                            "Le titre raconte CE fait, avec les détails exacts de la source.\n\n" + source_block)
        focus_text = " / ".join(focus or []) or source.get("title", "")
        user = self.fill(sections["user"], source_block=source_block, focus=focus_text, **title_vars)
        with self.llm_stage("titre"):
            data = self.llm.call_json_with_retry(
                [{"role": "system", "content": sections["system"]}, {"role": "user", "content": user}],
                temperature=1.0, max_completion_tokens=200, **self.simple_routing(),
            )
        title = str(data.get("title", "")).strip().strip('"«» ')
        reason = self.title_format_issue(title)
        if reason:
            print(f"   ↻ Titre rejeté ({reason}) : {title!r}")
            return None
        return title

    @staticmethod
    def title_format_issue(title: str) -> str | None:
        """Motif du rejet d'un titre sur sa forme (longueur, mots interdits), ou None s'il est acceptable."""
        if not title:
            return "titre vide"
        if len(title) > MAX_TITLE_CHARS:
            return f"{len(title)} caractères, maximum {MAX_TITLE_CHARS}"
        words = title_words(title)
        if words > 14:
            return f"{words} mots, maximum 14"
        banned = [w.strip() for w in BANNED_TITLE_WORDS if w in title.lower()]
        if banned:
            return f"mot interdit « {banned[0]} »"
        return None

    # ==========================================
    # HOOKS
    # ==========================================
    @staticmethod
    def _stats_text(previous_stats_list):
        if not previous_stats_list:
            return ""
        return ("STATS DES VIDÉOS PRÉCÉDENTES (inspire-toi de ce qui fonctionne) :\n"
                + "\n".join(f"- {s}" for s in previous_stats_list[:5]))

    @staticmethod
    def _hook_score(hook: dict) -> float | None:
        try:
            return float(hook.get("score"))
        except (TypeError, ValueError):
            return None

    def valid_hooks(self, hooks, topic) -> list[dict]:
        """Rejette les accroches notées 0/10, identiques au titre, trop longues, ou dont une date, un chiffre
        ou l'absence de tout élément précis ne vient pas de la source."""
        title = fold(topic)
        source = source_text(self._source, max_chars=100000) if self._source else ""
        kept = []
        for h in hooks:
            if not isinstance(h, dict) or not str(h.get("text", "")).strip():
                continue
            text = fold(h["text"])
            issues = claim_issues(h["text"], source, "accroche") if source else []
            if self._hook_score(h) == 0:
                print(f"   ↻ Accroche notée 0/10 rejetée : « {h['text']} »")
            elif text == title or (len(title) > 10 and title in text):
                print(f"   ↻ Accroche identique au titre rejetée : « {h['text']} »")
            elif count_words(h["text"]) > MAX_HOOK_WORDS:
                print(f"   ↻ Accroche de plus de {MAX_HOOK_WORDS} mots rejetée : « {h['text']} »")
            elif issues:
                print(f"   ↻ Accroche rejetée : {issues[0]}")
            else:
                kept.append(h)
        return kept

    def rate_hooks(self, hooks: list[dict], topic: str) -> list[dict]:
        """Notation par un gros modèle (brain.script_models) : phrase complète avec un verbe conjugué, et
        note de 1 à 10. Les accroches sans verbe conjugué sont rejetées ; la note remplace celle du générateur.
        Gros modèles indisponibles : RuntimeError (jamais de notation par un petit modèle)."""
        if not hooks:
            return []
        sections = self.read_sections("hook_rating.md")
        listing = "\n".join(f"{i}. {h['text']}" for i, h in enumerate(hooks))
        user = self.fill(sections["user"], topic=topic, hooks_list=listing)
        with self.llm_stage("accroches (notation)"):
            data = self.llm.call_json_with_retry(
                [{"role": "system", "content": sections["system"]}, {"role": "user", "content": user}],
                temperature=0, max_completion_tokens=800, **self.script_routing())
        notes = {}
        for note in data.get("notes") or []:
            try:
                notes[int(note.get("index"))] = note
            except (TypeError, ValueError, AttributeError):
                continue
        rated = []
        for i, h in enumerate(hooks):
            note = notes.get(i)
            if not note or note.get("phrase_complete") is not True or not note.get("verbe"):
                print(f"   ↻ Accroche sans verbe conjugué rejetée : « {h['text']} »")
                continue
            rated.append({**h, "score": self._hook_score(note) or 0, "verbe": note.get("verbe")})
        return [h for h in rated if h["score"] != 0]

    def generate_hook_variants(self, topic, n=5, previous_stats_list=None):
        """Accroches valables uniquement : une liste vide est régénérée (HOOK_ROUNDS fois), jamais
        remplacée par le titre."""
        sections = self.read_sections("hook.md")
        user = self.fill(sections["user"], n=n, topic=topic, grounding_block=self._grounding(),
                         stats_block=self._stats_text(previous_stats_list))
        for round_ in range(1, HOOK_ROUNDS + 1):
            with self.llm_stage("accroches"):
                data = self.llm.call_json_with_retry(
                    [{"role": "system", "content": self.fill(sections["system"], n=n)},
                     {"role": "user", "content": user}],
                    temperature=1.0, max_completion_tokens=1500, **self.simple_routing(),
                )
            hooks = self.rate_hooks(self.valid_hooks(data.get("hooks", []), topic), topic)
            if hooks:
                return sorted(hooks, key=lambda h: -(self._hook_score(h) or 0))[:n]
            print(f"   ↻ Aucune accroche valable ({round_}/{HOOK_ROUNDS}), nouvelle génération.")
            user += ("\n\nCORRECTION : tes accroches précédentes étaient identiques au titre, trop longues, sans "
                     "verbe conjugué, ou citaient une date ou un chiffre absent de la source. Écris des phrases "
                     "complètes, neuves, différentes du titre.")
        raise RuntimeError("aucune accroche valable après plusieurs générations.")

    # ==========================================
    # SCRIPT
    # ==========================================
    def _grounding(self) -> str:
        if not self._source:
            return ("Aucune source fournie : n'avance ni date, ni chiffre, ni nom "
                    "que tu ne peux pas garantir.")
        block = f'SOURCE VÉRIFIÉE :\n"""{source_text(self._source)}"""'
        if self._source.get("focus"):
            block += f"\n\nSUJET PRÉCIS : {' / '.join(self._source['focus'])}"
        if self._source.get("facts"):
            block += "\n\nFAITS UTILISABLES (chaque phrase du script s'appuie sur l'un d'eux) :\n" + \
                "\n".join(f"- {f}" for f in self._source["facts"])
        if self._source.get("answer"):
            block += f"\n\nRÉPONSE À LA QUESTION DU SUJET (à donner dans le script) : {self._source['answer']}"
        return block

    def generate_script(self, topic, chosen_hook=None):
        """Script ; tout échec compte dans le rapport (et, côté pipeline, pour l'abandon du sujet en attente)."""
        try:
            return super().generate_script(topic, chosen_hook=chosen_hook)
        except Exception:
            self._report["rejet : script"] += 1
            raise

    # Nombre de scènes : entre brain.min_scenes et cible + 1, accepté et renuméroté ; hors de cette plage, la
    # cible est réduite (cible, cible - 1, ...) au lieu de redemander le même nombre ou d'abandonner le sujet.
    reduce_target_on_count_mismatch = True

    def scene_count_accepted(self, count: int, target: int) -> bool:
        return self.min_scenes() <= count <= target + 1

    def scene_count_candidates(self, target: int) -> list[int]:
        return list(range(target, self.min_scenes() - 1, -1))[:3] or [max(target, self.min_scenes())]

    def run_report(self) -> list[str]:
        """Fin de run : sujets testés, retenus via le repli, rejets par motif."""
        if not self._report:
            return []
        rejects = sorted(((k.split(" : ", 1)[1], v) for k, v in self._report.items() if k.startswith("rejet")),
                         key=lambda kv: -kv[1])
        line = f"🧾 Sujets mystery : {self._report['sujets testés']} testé(s)"
        if self._report["repli"]:
            line += f", {self._report['repli']} passage(s) en repli"
        line += " ; rejets : " + (", ".join(f"{k} {v}" for k, v in rejects) if rejects else "aucun")
        return [line]

    def min_scenes(self) -> int:
        return int(self.config.get("brain", {}).get("min_scenes", DEFAULT_MIN_SCENES))

    def target_scene_count(self) -> int:
        """Matière courte (brain.short_video.max_facts faits au plus) : vidéo courte plutôt qu'un script étiré
        jusqu'à la durée visée, jamais sous brain.min_scenes."""
        count = super().target_scene_count()
        short = dict(self.config.get("brain", {}).get("short_video") or {})
        facts = len((self._source or {}).get("facts") or [])
        if short and facts and facts <= int(short.get("max_facts", 6)):
            per_scene = float(self.config.get("brain", {}).get("seconds_per_scene", 8))
            short_count = max(int(short.get("min_scenes", 3)), self.min_scenes(),
                              round(float(short.get("seconds", 24)) / per_scene))
            if short_count < count:
                print(f"⏱️ Matière courte ({facts} faits) : {short_count} scènes, environ "
                      f"{short.get('seconds', 24)} s, plutôt que {count}.")
                return short_count
        return count

    def build_script_prompt(self, topic, scene_count, chosen_hook, correction_feedback=""):
        hook = chosen_hook.get("text", "") if isinstance(chosen_hook, dict) else (chosen_hook or "")
        sections = self.read_sections("script.md")

        # {scene_count} est dans le prompt système : on le remplit aussi, et on y ajoute les règles de véracité
        system = self.fill(sections["system"], scene_count=scene_count)
        system += "\n\n" + self.read_prompt("veracity.md")

        user = self.fill(
            sections["user"],
            topic=topic, scene_count=scene_count, chosen_hook=hook,
            grounding_block=self._grounding(),
            rehook_scene=-(-scene_count // 3),      # tiers du script, arrondi au supérieur
            min_specific=self.min_specific(scene_count),
            max_sentence_words=self.style_cfg()["max_sentence_words"],
            banned_words=", ".join(f"« {w} »" for w in self.style_cfg()["banned_words"]) or "aucun",
        )
        if correction_feedback:
            user += "\n\n" + correction_feedback
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def style_cfg(self) -> dict:
        """brain.style (YAML) : longueur des phrases et adjectifs décoratifs interdits."""
        cfg = dict(self.config.get("brain", {}).get("style") or {})
        cfg.setdefault("max_sentence_words", 12)
        cfg.setdefault("banned_words", [])
        return cfg

    def min_specific(self, scene_count: int) -> int:
        """Scènes 'specific' exigées : brain.min_specific_ratio (2/3 par défaut, soit 4 sur 6)."""
        ratio = float(self.config.get("brain", {}).get("min_specific_ratio", 2 / 3))
        return min(scene_count, math.ceil(scene_count * ratio - 1e-9))

    # ==========================================
    # VALIDATION
    # ==========================================
    def check_style(self, scenes) -> list[str]:
        """Phrases trop longues et adjectifs décoratifs : renvoyés au LLM comme corrections."""
        cfg = self.style_cfg()
        limit = int(cfg["max_sentence_words"])
        banned = banned_word_pattern(cfg["banned_words"])
        issues = []
        for scene in scenes:
            for sentence in sentences(scene.get("text", "")):
                n = count_words(sentence)
                if n > limit:
                    issues.append(f"scène {scene['id']} : phrase de {n} mots (maximum {limit}) : « {sentence} ». "
                                  "Coupe-la en deux phrases courtes, comme à l'oral.")
            hits = sorted({m.group(0).lower() for m in banned.finditer(scene.get("text", ""))}) if banned else []
            if hits:
                issues.append(f"scène {scene['id']} : adjectif(s) décoratif(s) interdit(s) {', '.join(hits)}. "
                              "Supprime-les ou remplace-les par un fait précis.")
        return issues

    def check_specific_count(self, scenes) -> list[str]:
        scored = [s for s in scenes if s.get("role") != "cta"]
        need = self.min_specific(len(scored))
        have = sum(1 for s in scored if s.get("scene_type") == "specific")
        if have < need:
            return [f"{have} scène(s) 'specific' sur {len(scored)}, il en faut au moins {need} : montre le lieu "
                    "réel (bâtiment, salle, objet, détail) plutôt qu'une ambiance."]
        return []

    def validate_headline(self, data) -> None:
        """Bandeau rejeté s'il n'a ni verbe ni chiffre, ou s'il n'aligne que des noms de lieux."""
        headline = str(data.get("on_screen_title") or "").strip()
        if not headline:
            raise ValueError("champ 'on_screen_title' manquant.")
        words = [w for w in fold(headline.replace("|", " ")).split() if w not in STOPWORDS]
        places = set()
        for name in [s.get("location_name") for s in data.get("scenes", [])] + [
                self.last_pick.get(k) for k in ("source_title", "commune", "department", "region")]:
            places.update(fold(re.sub(r"\([^)]*\)", " ", str(name or ""))).split())
        if self._source:
            issues = claim_issues(headline.replace("|", " "), source_text(self._source, max_chars=100000), "bandeau")
            if issues:
                raise ValueError("on_screen_title " + issues[0])
        if words and all(w in places for w in words):
            raise ValueError(f"on_screen_title « {headline} » : seulement des noms de lieux. Le bandeau doit dire "
                             "ce qui s'est passé (un verbe ou un chiffre), pas seulement où.")
        if any(ch.isdigit() for ch in headline) or any(w in NUMBER_WORDS for w in words):
            return
        if not self._headline_has_verb(headline):
            raise ValueError(f"on_screen_title « {headline} » : ni verbe ni chiffre. Le bandeau doit contenir un "
                             "verbe conjugué ou un participe (« VENDUE », « DISPARAÎT ») ou un chiffre.")

    def _headline_has_verb(self, headline: str) -> bool:
        """Le LLM dit si le bandeau contient un verbe (conjugué, participe ou infinitif). En cas d'erreur
        du LLM, le bandeau est rejeté (le script est alors régénéré)."""
        prompt = (f"Bandeau : « {headline} »\nCe texte contient-il au moins un verbe (forme conjuguée, participe "
                  "passé ou présent, ou infinitif) ? Un nom (« construction », « vente ») n'est pas un verbe. "
                  'Réponds uniquement en JSON : {"verbe": true|false, "mot": "..."}')
        try:
            with self.llm_stage("bandeau"):
                data = self.llm.call_json_with_retry([{"role": "user", "content": prompt}],
                                                     temperature=0, max_completion_tokens=100, **self.simple_routing())
        except (RuntimeError, ValueError) as e:
            print(f"⚠️ Contrôle du bandeau impossible : {e}")
            return False
        return data.get("verbe") is True

    def validate_script_specifics(self, data, topic):
        if len(data["scenes"]) < self.min_scenes():      # la tolérance de comptage ne passe pas sous le plancher
            raise SceneCountError(f"{len(data['scenes'])} scènes : il en faut au moins {self.min_scenes()}.")
        if not str(data.get("closing_cta", "")).strip():
            raise ValueError("champ 'closing_cta' manquant.")
        self.validate_headline(data)
        self.locate_reveal(data)

        for scene in data["scenes"]:
            sid = scene["id"]
            words = len(str(scene["text"]).split())
            if words > MAX_WORDS_PER_SCENE:
                raise ValueError(f"scène {sid} : {words} mots, maximum {MAX_WORDS_PER_SCENE} "
                                 "(deux phrases courtes).")
            if scene.get("location_name") and not str(scene.get("location_country", "")).strip():
                raise ValueError(f"scène {sid} : 'location_country' manquant pour '{scene['location_name']}'.")
            # Le LLM confond parfois scene_type et role ("intro", "hook"...) : on le recalcule
            if scene.get("scene_type") not in SCENE_TYPES:
                scene["scene_type"] = "specific" if scene.get("location_name") else "generic"
            if scene.get("mood") not in MOODS:
                scene["mood"] = "intriguing"
            if scene.get("role") not in ROLES:
                scene["role"] = "context"

    def locate_reveal(self, data) -> None:
        """Phrase de la révélation (la mascotte « surpris » est montrée pendant cette phrase), choisie par le
        code et jamais par le LLM : la phrase du script la plus proche de la réponse à la question du sujet
        (similarité après normalisation) ; sous le seuil, la dernière phrase de la scène « reveal » (à défaut,
        de l'avant-dernière scène). Ne fait jamais échouer le script."""
        scenes = [s for s in data.get("scenes", []) if isinstance(s, dict)]
        for scene in scenes:
            scene.pop("reveal_sentence", None)
        if not scenes:
            return
        answer = str((self._source or {}).get("answer") or "").strip()
        best = None
        if answer:
            candidates = [(similarity(sentence, answer), i, sentence) for i, scene in enumerate(scenes)
                          if scene.get("role") not in ("hook", "cta") for sentence in sentences(scene.get("text", ""))]
            if candidates:
                score, index, sentence = max(candidates, key=lambda c: (c[0], c[1]))
                if score >= REVEAL_MIN_SIMILARITY:
                    best = (index, sentence)
        if best is None:
            index = next((i for i, s in enumerate(scenes) if s.get("role") == "reveal"), max(len(scenes) - 2, 0))
            parts = sentences(scenes[index].get("text", "")) or [str(scenes[index].get("text", ""))]
            best = (index, parts[-1])
        scenes[best[0]]["reveal_sentence"] = best[1]
        data["reveal_sentence"] = best[1]

    def check_script_logic(self, topic, data):
        """Lieux, style, scènes 'specific' puis fact-check : le fact-check tourne toujours (véracité)."""
        scenes = data["scenes"]
        writing = self.check_style(scenes) + self.check_specific_count(scenes)
        return self._check_geo(scenes) + writing + self._llm_factcheck(topic, scenes, data)

    def _check_geo(self, scenes):
        """Tous les lieux doivent être dans le pays attendu (brain.country dans le YAML)."""
        expected = str(self.config.get("brain", {}).get("country", "")).strip().lower()
        if not expected:
            return []
        issues = []
        for s in scenes:
            country = str(s.get("location_country") or "").strip().lower()
            if s.get("location_name") and country and country != expected:
                issues.append(f"scène {s['id']} : lieu « {s['location_name']} » en {s['location_country']}, "
                              f"attendu : {self.config['brain']['country']}.")
        return issues

    @staticmethod
    def _scenes_text(scenes):
        # texte intégral : résumer ferait disparaître les faits à vérifier
        return "\n".join(f"[scène {s['id']}] {s['text']}" for s in scenes)

    def _llm_factcheck(self, topic, scenes, data):
        text = self._scenes_text(scenes)
        grounded = bool(self._source and self._source.get("extract"))
        if grounded:
            sec = self.read_sections("factcheck_grounded.md")
            facts = self._source.get("facts") or []
            user = self.fill(sec["user"], source_title=self._source["title"],
                             source_extract=source_text(self._source), scenes_text=text, topic=topic,
                             facts_block="\n".join(f"- {f}" for f in facts) or "(aucune liste : la source entière)",
                             answer=self._source.get("answer") or "non précisée")
        else:
            sec = self.read_sections("factcheck_ungrounded.md")
            user = self.fill(sec["user"], topic=topic, scenes_text=text)

        try:
            with self.llm_stage("véracité"):
                res = self.llm.call_json_with_retry(
                    [{"role": "system", "content": sec["system"]}, {"role": "user", "content": user}],
                    temperature=0, max_completion_tokens=500, **self.script_routing(),   # véracité : gros modèles
                )
        except Exception as e:
            if str(self.config.get("brain", {}).get("factcheck", "")).lower() == "strict":
                # véracité stricte : un script non vérifié n'est jamais accepté (le sujet reste en attente)
                raise RuntimeError(f"contrôle de véracité indisponible : {e}") from e
            print(f"⚠️ Fact-check indisponible : {e}")
            data["factcheck"] = "skipped"
            return []

        # sans source, un LLM ne repère que les contradictions internes : à traiter comme "non vérifié"
        data["factcheck"] = "grounded" if grounded else "ungrounded"
        # seule la liste 'issues' fait foi (is_consistent peut la contredire)
        # avertissements de véracité : jamais ignorés (à la limite d'essais, le sujet est abandonné)
        return [VeracityIssue(str(i).strip()) for i in res.get("issues", []) if str(i).strip()]
