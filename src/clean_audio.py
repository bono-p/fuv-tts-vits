"""
Nettoyage audio specifique au TTS : coupe le silence en trop en debut/fin
de chaque clip, normalise le volume, et regenere un metadata_tts.json
qui pointe vers les fichiers nettoyes.

Ne modifie JAMAIS les fichiers originaux (utilises par le projet STT) :
tout est ecrit dans un dossier separe (data_config.clean_audio_subdir).

Usage :
    python clean_audio.py
"""

import json
import os

import librosa
import numpy as np
import soundfile as sf

from config import data_config


def clean_one(src_path: str, dst_path: str) -> bool:
    """Coupe le silence, normalise le volume, ecrit un nouveau wav."""
    try:
        audio, sr = librosa.load(src_path, sr=data_config.sampling_rate, mono=True)
    except Exception as e:
        print(f"  [X] Impossible de lire {os.path.basename(src_path)} : {e}")
        return False

    if len(audio) == 0:
        print(f"  [X] Fichier vide ignore : {os.path.basename(src_path)}")
        return False

    # 1. Coupe le silence au-dela du seuil top_db
    trimmed, _ = librosa.effects.trim(audio, top_db=data_config.top_db)

    if len(trimmed) == 0:
        # Le fichier est probablement tout en silence / bruit constant
        print(f"  [!] {os.path.basename(src_path)} semble entierement silencieux, ignore.")
        return False

    # 2. Rajoute une petite marge de silence propre en debut/fin
    pad_samples = int(data_config.silence_padding_sec * sr)
    padded = np.concatenate([
        np.zeros(pad_samples, dtype=trimmed.dtype),
        trimmed,
        np.zeros(pad_samples, dtype=trimmed.dtype),
    ])

    # 3. Normalise le volume au niveau de crete cible
    peak = np.max(np.abs(padded))
    if peak > 0:
        target_peak_linear = 10 ** (data_config.target_peak_dbfs / 20)
        padded = padded * (target_peak_linear / peak)

    sf.write(dst_path, padded, sr, subtype="PCM_16")
    return True


def main():
    audio_dir = os.path.join(data_config.drive_root, data_config.audio_subdir)
    metadata_path = os.path.join(data_config.drive_root, data_config.metadata_filename)
    clean_audio_dir = os.path.join(data_config.drive_root, data_config.clean_audio_subdir)
    clean_metadata_path = os.path.join(data_config.drive_root, data_config.clean_metadata_filename)

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.json introuvable : {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    os.makedirs(clean_audio_dir, exist_ok=True)

    print(f"[i] {len(records)} enregistrement(s) a nettoyer...")

    clean_records = []
    skipped = 0

    for rec in records:
        fname = rec.get("file_name")
        text = rec.get("text", "").strip()
        if not fname or not text:
            skipped += 1
            continue

        src_path = os.path.join(audio_dir, fname)
        if not os.path.isfile(src_path):
            skipped += 1
            continue

        dst_path = os.path.join(clean_audio_dir, fname)
        if clean_one(src_path, dst_path):
            clean_records.append({"file_name": fname, "text": text})
        else:
            skipped += 1

    with open(clean_metadata_path, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] {len(clean_records)} fichier(s) nettoye(s) avec succes.")
    if skipped:
        print(f"[!] {skipped} fichier(s) ignore(s) (illisible, vide, ou manquant).")
    print(f"[OK] Audio nettoye dans : {clean_audio_dir}")
    print(f"[OK] Metadata TTS ecrit dans : {clean_metadata_path}")


if __name__ == "__main__":
    main()
