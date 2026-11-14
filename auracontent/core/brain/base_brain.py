import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auracontent.core.brain.llm_client import LLMClient
from auracontent.utils.text_utils import _title_tokens

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class VeracityIssue(str):
    """Avertissement du contrôle de véracité (fait absent de la source, phrase sans fait) : jamais ignoré.

    À la limite d'essais, le script n'est pas accepté : le sujet est abandonné (TopicRejected)."""


class TopicRejected(RuntimeError):
    """Le sujet ne peut pas donner un script véridique : on l'abandonne et on passe au suivant."""


class SceneCountError(ValueError):
    """Le LLM n'a pas tenu le nombre de scènes demandé (une chaîne peut alors abandonner le sujet)."""
_SECTION_HEADER = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def rate_limit_routing_from(brain_cfg: dict | None) -> dict:
    """`brain.rate_limit` du YAML (« wait » ou « fallback ») ; absent = rien (comportement par défaut du client)."""
    value = (brain_cfg or {}).get("rate_limit")
    return {"rate_limit": str(value)} if value else {}


def simple_routing_from(brain_cfg: dict | None) -> dict:
    """Routage des tâches simples d'après la section `brain` du YAML (voir BaseBrain.simple_routing)."""
    b = brain_cfg or {}
    if not b.get("simple_models"):
        return rate_limit_routing_from(b)
    excluded = b.get("simple_fallback_excluded", b.get("script_models") or ["gemini"])
    return {"allowed_models": list(b["simple_models"]), "fallback_excluded": list(excluded),
            **rate_limit_routing_from(b)}


@dataclass
class BaseBrain:
    config: dict[str, Any]
    llm: LLMClient
    channel_name: str = "base"

    # ==========================================
    # PROMPTS (fichiers .md de la chaîne)
    # ==========================================

    @property
    def prompts_dir(self) -> Path:
        raise NotImplementedError("La sous-classe doit définir prompts_dir.")

    def read_prompt(self, name: str) -> str:
        return (self.prompts_dir / name).read_text(encoding="utf-8").strip()

    def read_sections(self, name: str) -> dict[str, str]:
        """Découpe un .md en sections '# Titre' -> {'titre en minuscules': contenu}."""
        text = self.read_prompt(name)
        headers = list(_SECTION_HEADER.finditer(text))
        sections = {}
        for i, h in enumerate(headers):
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            sections[h.group(1).strip().lower()] = text[h.end():end].strip()
        missing = {"system", "user"} - sections.keys()
        if missing:
            raise KeyError(f"{name} : section(s) manquante(s) {sorted(missing)} (attendu '# System' et '# User').")
        return sections

    @staticmethod
    def fill(template: str, **values) -> str:
        """Remplace {cle} par sa valeur. Les accolades inconnues (exemples JSON) restent intactes."""
        return _PLACEHOLDER.sub(
            lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template
        )

    # ==========================================
    # MÉTHODES GÉNÉRIQUES (Communes à tous)
    # ==========================================

    def is_duplicate_topic_llm(self, candidate, history):
        """Vérifie si le sujet raconte la même histoire qu'un sujet déjà publié."""
        if not history:
            return False, None
        recent = history[-15:]
        # Filtre local gratuit : sans au moins 2 mots significatifs (> 3 lettres) en commun avec un sujet
        # récent, ce ne peut pas être la même histoire -> pas d'appel LLM
        cand = {t for t in _title_tokens(candidate) if len(t) > 3}
        overlap = max((len(cand & {t for t in _title_tokens(h) if len(t) > 3}) for h in recent), default=0)
        if overlap < 2:
            return False, None
        prompt = f"""Voici un nouveau sujet de vidéo : "{candidate}"

Voici les 15 derniers sujets déjà publiés :
{chr(10).join(f"- {t}" for t in recent)}

Le nouveau sujet raconte-t-il le MÊME événement, la même personne, le même objet ou le même lieu
qu'un des sujets déjà publiés ? Un simple thème commun (mystère, disparition, histoire de France)
ne suffit PAS à en faire un doublon.
Réponds uniquement en JSON : {{"is_duplicate": true/false, "matched_topic": "..." ou null}}"""

        with self.llm_stage("doublon"):
            data = self.llm.call_json_with_retry(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=300,
                **self.simple_routing(),
            )
        if data.get("is_duplicate"):
            return True, data.get("matched_topic")
        return False, None

    def generate_hook_variants(self, topic, n=5, previous_stats_list=None):
        """Génère N variantes d'accroches de façon générique."""
        prompt = f"""SUJET:\n{topic}\n\nGENERE {n} hooks viraux en francais.\n
        RETURNS JSON:
        {{
          "hooks": [
            {{"text": "...", "pattern": "question", "raison": "..."}}
          ]
        }}"""

        messages = [
            {"role": "system", "content": "Tu produis uniquement du JSON valide sans texte autour."},
            {"role": "user", "content": prompt},
        ]

        with self.llm_stage("accroches"):
            data = self.llm.call_json_with_retry(messages, temperature=1.1, max_completion_tokens=2000,
                                                 **rate_limit_routing_from(self.config.get("brain")))
        hooks = [h for h in data.get("hooks", []) if isinstance(h, dict) and h.get("text")]
        return hooks[:n] or [{"text": topic, "pattern": "default", "raison": ""}]

    def target_scene_count(self) -> int:
        """Nombre de scènes déduit de la durée visée (env VIDEO_DURATION ou brain.target_duration).

        La narration Gemini tourne autour de 8 s par scène : 45 s -> 6 scènes. `scene_count` reste le plafond.
        """
        brain_cfg = self.config.get("brain", {})
        max_scenes = int(brain_cfg.get("scene_count", 11))
        try:
            duration = float(os.getenv("VIDEO_DURATION") or brain_cfg.get("target_duration", 45))
        except ValueError:
            duration = 45.0
        per_scene = float(brain_cfg.get("seconds_per_scene", 8))
        return max(5, min(max_scenes, round(duration / per_scene)))

    def script_routing(self) -> dict:
        """brain.script_models (YAML) : modèles autorisés pour le script ; absent = tous (autres chaînes)."""
        b = self.config.get("brain", {})
        allowed = b.get("script_models")
        if not allowed:
            return rate_limit_routing_from(b)
        return {"allowed_models": list(allowed), "wait_rounds": int(b.get("script_wait_rounds", 3)),
                "wait_seconds": float(b.get("script_wait_seconds", 60)), **rate_limit_routing_from(b)}

    def simple_routing(self) -> dict:
        """Tâches simples (choix de page, titre, contrôles, accroches) : brain.simple_models (petits modèles).

        Si tous échouent, repli sur les autres modèles SAUF ceux du script (brain.script_models, ou
        brain.simple_fallback_excluded) : Gemini et les gros modèles restent réservés au script.
        Absent du YAML = tous les modèles (autres chaînes).
        """
        return simple_routing_from(self.config.get("brain", {}))

    def generate_script(self, topic, chosen_hook=None):
        """Point d'entrée standard pour générer un script."""
        target_scenes = self.target_scene_count()
        print(f"⏱️ {target_scenes} scènes visées")
        # brain.script_max_retries : corrections demandées au LLM après le premier essai (absent = 2)
        retries = int(self.config.get("brain", {}).get("script_max_retries", 2))
        return self.generate_script_with_target(
            topic, scene_count=target_scenes, chosen_hook=chosen_hook, max_retries=retries
        )

    def generate_script_with_target(self, topic, scene_count=11, chosen_hook=None, max_retries=2):
        """
        Gère la boucle de repli sur le nombre de scènes : si le LLM échoue (timeout/tokens),
        on retente avec moins de scènes (ex: 11 -> 8 -> 5).
        """
        candidate_counts = self.scene_count_candidates(scene_count)

        last_error = None
        for idx, candidate_count in enumerate(candidate_counts):
            if idx > 0:
                print(f"⚠️ Nouvelle tentative avec {candidate_count} scènes pour respecter le budget.")
            try:
                with self.llm_stage("script"):
                    return self._generate_script_attempt(topic, candidate_count, chosen_hook, max_retries,
                                                         can_reduce=idx < len(candidate_counts) - 1)
            except TopicRejected:
                raise                     # moins de scènes ne rendrait pas le script plus vrai
            except (RuntimeError, ValueError) as e:
                last_error = e
                print(f"❌ Échec de génération avec {candidate_count} scènes : {e}")

        raise RuntimeError(f"Impossible de générer un script. Dernière erreur : {last_error}") from last_error

    def _generate_script_attempt(self, topic, scene_count, chosen_hook, max_retries, can_reduce=False):
        """Boucle de correction (validation de structure puis fact-check)."""
        correction_feedback = ""
        for attempt in range(max_retries + 1):
            last = attempt == max_retries
            messages = self.build_script_prompt(topic, scene_count, chosen_hook, correction_feedback)
            data = self.llm.call_json_with_retry(
                messages,
                temperature=0.7,
                max_completion_tokens=min(scene_count * 350 + 1200, 6500),
                **self.script_routing(),
            )

            scenes = data.get("scenes", [])
            if not self.scene_count_accepted(len(scenes), scene_count):
                if self.reduce_target_on_count_mismatch and can_reduce:
                    # hors de la plage acceptée : cible réduite (scene_count_candidates) plutôt que de redemander ;
                    # à la dernière cible, correction demandée au même nombre (jamais un seul essai)
                    raise SceneCountError(f"{len(scenes)} scène(s) rendue(s) pour {scene_count} visée(s).")
                if last:
                    raise SceneCountError(f"Échec du comptage des scènes après {max_retries + 1} tentatives.")
                correction_feedback = (f"CORRECTION OBLIGATOIRE : génère EXACTEMENT {scene_count} scènes. "
                                       f"Tu en as fait {len(scenes)}.")
                continue
            if len(scenes) != scene_count:
                print(f"ℹ️ {len(scenes)} scènes au lieu de {scene_count} : accepté et renuméroté.")
                for idx, scene in enumerate(scenes, start=1):   # numérotation continue
                    if isinstance(scene, dict):
                        scene["id"] = idx

            try:
                self._validate_script_base(data)
                self.validate_script_specifics(data, topic)
            except ValueError as e:
                if last:
                    raise
                correction_feedback = f"CORRECTION OBLIGATOIRE : {e}"
                continue

            issues = self.check_script_logic(topic, data)
            if issues:
                if not last:
                    correction_feedback = "CORRECTIONS OBLIGATOIRES :\n" + "\n".join(f"- {i}" for i in issues)
                    continue
                blocking = [i for i in issues if isinstance(i, VeracityIssue)]
                if blocking:
                    raise TopicRejected("contrôle de véracité en échec après "
                                        f"{max_retries + 1} essais : " + " ; ".join(blocking))
                if self.config.get("brain", {}).get("style_warnings") == "reject":
                    # brain.style_warnings: reject -> jamais de script publié avec des avertissements
                    raise ValueError(f"avertissements non résolus après {max_retries + 1} essais : "
                                     + " ; ".join(issues))
                print("⚠️ Script généré avec des avertissements de style ignorés (limite d'essais atteinte) :")
                for i in issues:
                    print(f"   - {i}")
                data["unresolved_issues"] = issues
            return data

    def _validate_script_base(self, data):
        """Applique les valeurs par défaut et vérifie les clés critiques."""
        for idx, scene in enumerate(data.get("scenes", []), start=1):
            if not isinstance(scene, dict) or not scene.get("text"):
                raise ValueError(f"scène {idx} : champ 'text' manquant.")
            scene.setdefault("id", idx)
            scene.setdefault("pause_after_ms", 800)
            scene.setdefault("voice_type", "narrator")
            scene.setdefault("scene_type", "generic")
        if not str(data.get("title", "")).strip():
            data["title"] = "Titre généré"

    # ==========================================
    # COMPTAGE DES SCÈNES ET ÉTAPES (surchargeables)
    # ==========================================
    reduce_target_on_count_mismatch = False   # True : nombre hors plage -> cible réduite, sans correction

    def scene_count_accepted(self, count: int, target: int) -> bool:
        """Nombre de scènes rendu accepté (puis renuméroté) : brain.scene_count_tolerance autour de la cible."""
        tolerance = int(self.config.get("brain", {}).get("scene_count_tolerance", 0))
        return abs(count - target) <= tolerance

    def scene_count_candidates(self, target: int) -> list[int]:
        """Cibles essayées dans l'ordre si un essai échoue (11 -> 8 -> 5)."""
        counts, sc = [], target
        while sc >= 5 and len(counts) < 3:
            counts.append(sc)
            sc -= 3
        return counts or [target]

    def llm_stage(self, name: str):
        """Étape nommée pour le rapport des appels LLM (sans effet avec un client qui ne la gère pas)."""
        stage = getattr(self.llm, "in_stage", None)
        return stage(name) if callable(stage) else nullcontext()

    # ==========================================
    # MÉTHODES À SURCHARGER
    # ==========================================

    def pick_topic(self):
        """Retourne un sujet inédit. Doit gérer lui-même l'anti-doublon (is_duplicate_topic_llm)."""
        raise NotImplementedError

    def build_script_prompt(self, topic, scene_count, chosen_hook, correction_feedback):
        """Retourne la liste 'messages' pour l'API."""
        raise NotImplementedError

    def validate_script_specifics(self, data, topic):
        """Règles de validation propres à la chaîne. Lever ValueError pour demander une correction."""

    def check_script_logic(self, topic, data):
        """Liste des erreurs factuelles/logiques. Vide = script validé."""
        return []

    def build_visuals(self, script: dict, scenes: list[dict], work_dir: Path, video_cfg: dict) -> list | None:
        """Visuels fabriqués par la chaîne elle-même (scènes dessinées par code), appelés après la voix :
        chaque scène connaît alors sa durée réelle (`duration`). Retourne un chemin de clip par scène, ou None
        pour garder la recherche d'images habituelle (AssetManager). Une scène peut recevoir `sfx_cues`."""
        return None
