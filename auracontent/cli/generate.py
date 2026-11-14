"""Point d'entrée de la génération. À lancer depuis la racine du dépôt :

    python -m auracontent.cli.generate                # toutes les chaînes du registre, script + vidéo
    python -m auracontent.cli.generate mystery              # une seule chaîne
    python -m auracontent.cli.generate mystery --script-only
    python -m auracontent.cli.generate mystery --upload     # dépose la vidéo + légende sur Hugging Face (CI)
    python -m auracontent.cli.generate mystery --sync-state # historiques (sujets, hooks) lus/sauvés sur HF (CI)
    python -m auracontent.cli.generate finance --script-only --allow-disabled  # essai local d'une chaîne coupée

Une chaîne marquée `enabled: false` dans son YAML est ignorée : aucun appel LLM, aucun historique lu ou écrit.
--allow-disabled la lance quand même, en local seulement (refusé avec --upload ou --sync-state).
"""
import argparse
import sys

from auracontent.channels.base_channel import channel_enabled
from auracontent.channels.registry import CHANNEL_REGISTRY
from auracontent.core.brain.global_brain import GlobalBrain
from auracontent.core.brain.llm_client import LLMClient
from auracontent.core.pipeline import produce_video
from auracontent.services.publishing.state_sync import pull_state, push_state
from auracontent.utils.pending_topic import PENDING_FILE


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="AuraContent : génération de vidéos courtes")
    parser.add_argument("channels", nargs="*", help="chaînes à produire (défaut : toutes)")
    parser.add_argument("--script-only", action="store_true", help="s'arrêter au script JSON")
    parser.add_argument("--upload", action="store_true", help="envoyer la vidéo finale sur Hugging Face")
    parser.add_argument("--sync-state", action="store_true", help="lire/sauver les historiques sur Hugging Face")
    parser.add_argument("--allow-disabled", action="store_true",
                        help="lancer aussi les chaînes `enabled: false` (essai local, sans --upload ni --sync-state)")
    args = parser.parse_args(argv)
    if args.allow_disabled and (args.upload or args.sync_state):
        parser.error("--allow-disabled est réservé aux essais locaux : incompatible avec --upload et --sync-state.")

    # chaînes de l'exécution : leurs historiques seulement (fichiers partagés ou dossiers isolés)
    channels = []
    for name in args.channels or list(CHANNEL_REGISTRY):
        if name in CHANNEL_REGISTRY and not channel_enabled(name) and not args.allow_disabled:
            print(f"⏭️ {name} : désactivée (`enabled: false` dans config/channels/{name}.yaml), ignorée.")
            continue
        channels.append(name)
    if not channels:
        print("💤 Aucune chaîne active à générer.")
        return 0
    if args.sync_state:
        pull_state(channels)

    llm = LLMClient.create_default()
    brain = GlobalBrain(llm_client=llm)

    results = {}
    for name in channels:
        try:
            script_path = brain.run_channel(name)
            if not args.script_only:
                produce_video(brain.channels[name], llm, script_path, upload=args.upload)
            results[name] = True
        except Exception as e:
            print(f"❌ {name} : {e}")
            results[name] = False

    for name in channels:                               # rapport propre à la chaîne (sujets testés, rejets)
        report = getattr(getattr(brain.channels.get(name), "brain", None), "run_report", None)
        if callable(report):
            for line in report():
                print(line)
    for line in llm.usage_report():
        print(line)
    if llm.usage:
        print(llm.budget_line(sum(results.values())))

    # On ne sauvegarde l'état que si au moins une chaîne a réussi (évite de « brûler » un sujet sans vidéo).
    # Après un échec, seul le sujet en attente est sauvé : l'exécution suivante le reprend.
    if args.sync_state:
        if any(results.values()):
            push_state(channels=channels)
        else:
            push_state((PENDING_FILE,), channels=channels)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
