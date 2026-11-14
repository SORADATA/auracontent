# auracontent/core/brain/global_brain.py
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from auracontent.channels.base_channel import BaseChannel
from auracontent.channels.registry import CHANNEL_REGISTRY
from auracontent.core.brain.llm_client import LLMClient
import os

from auracontent.services.publishing.zernio import get_latest_videos_stats
from auracontent.utils.hook_tracker import compute_pattern_scores, load_hook_history, select_hook
from auracontent.core.brain.base_brain import TopicRejected
from auracontent.utils.pending_topic import (
    clear_pending,
    load_pending,
    note_rejected,
    note_script_attempt,
    save_pending,
)
from auracontent.utils.topic_tracker import save_global_history
from config import settings


@dataclass
class GlobalBrain:
    llm_client: LLMClient
    output_dir: Path = settings.OUTPUT_DIR / "scripts"
    channels: dict[str, BaseChannel] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Instancie chaque chaîne du registre avec SA config YAML et le client LLM partagé."""
        self.channels = {
            name: BaseChannel(name, brain_cls, self.llm_client)
            for name, brain_cls in CHANNEL_REGISTRY.items()
        }
        print(f"🌍 GlobalBrain initialisé avec {len(self.channels)} chaîne(s) active(s).")

    # ==========================================
    # MÉTHODES D'ORCHESTRATION
    # ==========================================

    def run_channel(self, channel_name: str, previous_stats_list=None) -> Path:
        """Sujet -> hooks -> script pour une chaîne. Retourne le chemin du script JSON sauvegardé."""
        if channel_name not in self.channels:
            raise ValueError(f"Chaîne inconnue : {channel_name}. Vérifiez channels/registry.py.")
        channel = self.channels[channel_name]
        brain = channel.brain
        print(f"\n🚀 Démarrage de la chaîne : {channel_name.upper()}")

        # 0. Boucle de feedback : performances des dernières vidéos de la chaîne
        if previous_stats_list is None:
            previous_stats_list = self._fetch_stats(channel.section("publish"))

        b = channel.section("brain") or {}
        resumable = hasattr(brain, "export_state") and hasattr(brain, "restore_state")
        rounds = max(1, int(b.get("veracity_topic_rounds", 3)))
        for round_ in range(1, rounds + 1):
            topic, hooks, chosen_hook = self._topic_and_hook(channel_name, channel, brain, b, resumable,
                                                             previous_stats_list, resume=round_ == 1)
            # 3. Script (en cas d'échec, le sujet reste en attente pour la prochaine exécution, sauf s'il est
            # abandonné par le contrôle de véracité : on passe alors au sujet suivant)
            try:
                script = brain.generate_script(topic, chosen_hook=chosen_hook)
                break
            except TopicRejected as e:
                pick = dict(getattr(brain, "last_pick", {}) or {})
                print(f"🚫 Sujet abandonné ({e}) : {topic}")
                if resumable:
                    clear_pending(channel_name)
                note_rejected(channel_name, pick.get("source_title"), pick.get("anecdote_id"), str(e))
                if round_ == rounds:
                    raise RuntimeError(f"{rounds} sujet(s) abandonné(s) par le contrôle de véracité.") from e
                print(f"↪️ Sujet suivant ({round_ + 1}/{rounds}).")
            except Exception as e:
                if resumable:
                    failures = note_script_attempt(channel_name)
                    limit = int(b.get("resume_max_attempts", 3))
                    if failures >= limit:
                        # N-ième échec du script sur ce sujet : abandonné tout de suite et plus jamais proposé
                        pick = dict(getattr(brain, "last_pick", {}) or {})
                        print(f"🗑️ Sujet abandonné après {failures} échec(s) du script : {topic}")
                        clear_pending(channel_name)
                        note_rejected(channel_name, pick.get("source_title"), pick.get("anecdote_id"),
                                      f"{failures} échec(s) du script : {e}")
                    else:
                        print(f"💾 Sujet, titre et accroche gardés ({failures}/{limit} échec(s)) : la prochaine "
                              "exécution reprendra ce sujet.")
                raise
        pick = dict(getattr(brain, "last_pick", {}) or {})
        script.update({"topic": topic, "channel": channel_name, "hook": chosen_hook,
                       "hook_variants": hooks, "pick": pick})

        # 4. Sauvegarde du script puis de l'historique (seulement si tout a réussi)
        path = self._save_script(channel_name, script)
        save_global_history(topic, channel_name, script_path=str(path), **pick)
        if resumable:
            clear_pending(channel_name)
        print(f"✅ Script sauvegardé : {path}")
        return path

    def _topic_and_hook(self, channel_name, channel, brain, b, resumable, previous_stats_list, resume=True):
        """(sujet, accroches, accroche choisie) : reprise d'un sujet validé, sinon nouveau sujet."""
        pending = load_pending(channel_name, max_attempts=int(b.get("resume_max_attempts", 3)),
                               max_age_days=float(b.get("resume_max_days", 7))) if resumable and resume else None
        if pending:
            # Une chaîne peut refuser un sujet en attente devenu invalide (concept retiré du YAML, paramètres
            # hors limites) en levant ValueError/KeyError : il est abandonné et un nouveau sujet est choisi.
            try:
                brain.restore_state(pending["brain"])
            except (ValueError, KeyError) as e:
                print(f"🗑️ Sujet en attente abandonné ({e}) : {pending.get('topic')}")
                clear_pending(channel_name)
                pending = None
        if pending:
            # Reprise : sujet, titre et accroche déjà validés lors d'une exécution précédente
            topic, hooks, chosen_hook = pending["topic"], pending["hooks"], pending["chosen_hook"]
            print(f"♻️ Reprise du sujet validé ({pending.get('script_attempts', 0)} échec(s) de script) : {topic}")
            print(f"🪝 Accroche : {chosen_hook.get('text')}")
            return topic, hooks, chosen_hook
        # 1. Sujet (l'anti-doublon sur l'historique global est fait dans pick_topic)
        topic = brain.pick_topic()
        print(f"🎯 Sujet retenu : {topic}")

        # 2. Accroche choisie par le bandit (pattern le plus performant, 20 % d'exploration)
        hooks = brain.generate_hook_variants(topic, n=5, previous_stats_list=previous_stats_list)
        scores = compute_pattern_scores(previous_stats_list, load_hook_history(channel_name))
        if scores:
            print(f"📈 Scores des patterns : {scores}")
        chosen_hook = select_hook(hooks, pattern_scores=scores) or hooks[0]
        print(f"🪝 Accroche : {chosen_hook.get('text')}")
        if resumable:
            save_pending(channel_name, {"topic": topic, "hooks": hooks, "chosen_hook": chosen_hook,
                                        "brain": brain.export_state()})
        return topic, hooks, chosen_hook

    def run_all(self) -> dict[str, Path | None]:
        """Lance séquentiellement toutes les chaînes actives."""
        print("\n🌍 Lancement de la production multi-chaînes")
        results = {}
        for name in self.channels:
            try:
                results[name] = self.run_channel(name)
            except Exception as e:
                print(f"❌ Échec critique sur la chaîne {name} : {e}")
                results[name] = None
        return results

    @staticmethod
    def _fetch_stats(publish_cfg: dict):
        """Stats Zernio du compte de la chaîne (noms des variables dans la section `publish` du YAML)."""
        api_key = os.getenv(publish_cfg.get("zernio_key_env", "ZERNIO_API_KEY"))
        stats_cfg = publish_cfg.get("stats", {})
        account_envs = stats_cfg.get("account_envs")
        if account_envs:
            # stats.account_envs : uniquement les comptes de la chaîne (un appel par compte) ; un compte
            # sans variable est ignoré, jamais remplacé par « tous les comptes » de la clé
            ids = [os.getenv(env) for env in account_envs]
            if not any(ids):
                print("ℹ️ Aucun identifiant de compte de la chaîne : boucle de feedback désactivée.")
                return None
            merged = []
            for account_id in filter(None, ids):
                merged += get_latest_videos_stats(api_key, platform=stats_cfg.get("platform"),
                                                  account_id=account_id, limit=stats_cfg.get("limit", 20)) or []
            return merged or None
        account_env = stats_cfg.get("account_env")
        return get_latest_videos_stats(
            api_key,
            platform=stats_cfg.get("platform"),
            account_id=os.getenv(account_env) if account_env else None,
            limit=stats_cfg.get("limit", 20),
        )

    def _save_script(self, channel_name: str, script: dict) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{channel_name}_{stamp}.json"
        path.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
