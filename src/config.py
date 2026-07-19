"""
Configuration centrale du pipeline TTS fulfulde (fuv), voix unique.

Reutilise le meme dataset brut que le projet STT (fuv-stt-whisper),
mais produit une version "nettoyee" specifique au TTS (silence coupe,
volume normalise) sans toucher aux fichiers originaux.
"""

from dataclasses import dataclass


@dataclass
class DataConfig:
    # Meme dataset source que le STT (audio deja converti en .wav 16kHz mono)
    drive_root: str = "/content/drive/MyDrive/fuv-stt-dataset"
    audio_subdir: str = "audio"
    metadata_filename: str = "metadata.json"

    # Sortie du nettoyage audio specifique TTS (ne modifie pas l'original)
    clean_audio_subdir: str = "audio_tts_clean"
    clean_metadata_filename: str = "metadata_tts.json"

    sampling_rate: int = 16000

    # Marge de silence gardee au debut/fin de chaque clip (en secondes)
    silence_padding_sec: float = 0.15

    # Seuil de decoupe du silence (en dB sous le pic) — augmente si trop
    # de silence est coupe, diminue si du silence subsiste
    top_db: int = 30

    # Niveau cible de normalisation (crete a -1 dBFS = tres proche du
    # maximum sans ecreter)
    target_peak_dbfs: float = -1.0


@dataclass
class HubConfig:
    # Repo HF Hub prive ou sera pousse le dataset TTS assemble
    dataset_repo_id: str = "bonopassale/fuv-tts-monospeaker"

    # Checkpoint de base MMS-TTS pour le fulfulde (macrolangue "ful")
    base_model: str = "facebook/mms-tts-ful"
    language_code: str = "ful"

    # Checkpoint intermediaire avec discriminateur (cree une seule fois,
    # via convert_original_discriminator_checkpoint.py de finetune-hf-vits)
    converted_checkpoint_repo_id: str = "bonopassale/mms-tts-ful-with-discriminator"

    # Modele final fine-tune, pousse sur le Hub a la fin de l'entrainement
    finetuned_model_repo_id: str = "bonopassale/tts-fuv-voice"


data_config = DataConfig()
hub_config = HubConfig()
