"""
Assemble le dataset TTS (audio nettoye + texte) en un objet HuggingFace
`datasets.Dataset`, et le pousse sur le Hub (prive par defaut) pour que
le script de fine-tuning de finetune-hf-vits puisse le charger via
`load_dataset(dataset_name)`.

Prerequis : avoir execute clean_audio.py avant (ce script lit
metadata_tts.json + audio_tts_clean/).

Usage :
    python prepare_dataset.py
"""

import json
import os

import soundfile as sf
from datasets import Audio, Dataset

from config import data_config, hub_config


def _is_decodable(path: str) -> bool:
    """Verifie que le fichier audio est reellement lisible et non vide."""
    try:
        data, sr = sf.read(path)
    except Exception:
        return False
    return len(data) > 0


def load_clean_records() -> list:
    metadata_path = os.path.join(data_config.drive_root, data_config.clean_metadata_filename)
    audio_dir = os.path.join(data_config.drive_root, data_config.clean_audio_subdir)

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} introuvable. As-tu bien lance clean_audio.py avant ?"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    valid = []
    bad_files = []

    for rec in records:
        full_path = os.path.join(audio_dir, rec["file_name"])
        if not os.path.isfile(full_path):
            bad_files.append((rec["file_name"], "fichier introuvable"))
            continue
        if not _is_decodable(full_path):
            bad_files.append((rec["file_name"], "illisible ou vide"))
            continue
        valid.append({"audio": full_path, "text": rec["text"]})

    if bad_files:
        print(f"[!] {len(bad_files)} fichier(s) exclu(s) du dataset (corrompus ou vides) :")
        for fname, reason in bad_files[:20]:
            print(f"    - {fname} ({reason})")
        if len(bad_files) > 20:
            print(f"    ... et {len(bad_files) - 20} autres")
        print(
            "    -> Ces fichiers ne seront PAS publies. Verifie/re-enregistre-les "
            "si tu veux les recuperer plus tard."
        )

    if not valid:
        raise RuntimeError("Aucun enregistrement valide trouve dans le dataset nettoye.")

    return valid


def main():
    print("[i] Chargement des enregistrements nettoyes...")
    records = load_clean_records()
    print(f"[OK] {len(records)} paires audio/texte pretes.")

    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=data_config.sampling_rate))

    print(f"\n[i] Publication sur le Hub : {hub_config.dataset_repo_id} (prive)")
    dataset.push_to_hub(hub_config.dataset_repo_id, private=True)

    print(
        "\n[OK] Dataset publie. Utilise cet identifiant comme `dataset_name` "
        f"dans ton fichier de config d'entrainement : {hub_config.dataset_repo_id}"
    )


if __name__ == "__main__":
    main()
