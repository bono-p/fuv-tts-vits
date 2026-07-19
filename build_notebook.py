import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Fine-tuning MMS-TTS (VITS) pour le fulfulde — voix unique

Ce notebook fine-tune `facebook/mms-tts-ful` sur ta propre voix, en
reutilisant le meme dataset audio que le projet STT (fuv-stt-whisper).

Etapes :
1. Monter Google Drive
2. Cloner ce repo (`fuv-tts-vits`) + le repo d'entrainement `finetune-hf-vits`
3. Installer les dependances des deux repos
4. Se connecter au Hugging Face Hub (necessaire pour publier dataset + modele)
5. Nettoyer l'audio (silence + normalisation du volume)
6. Assembler et publier le dataset sur le Hub
7. Convertir le checkpoint MMS-TTS-ful (ajout du discriminateur, une seule fois)
8. Lancer le fine-tuning
9. Tester le modele obtenu

Pre-requis : le dataset `fuv-stt-dataset` (audio + metadata.json) doit deja
etre sur ton Drive, comme pour le projet STT."""
))

cells.append(nbf.v4.new_markdown_cell("## 1. Monter Google Drive"))
cells.append(nbf.v4.new_code_cell(
"""from google.colab import drive
drive.mount('/content/drive')"""
))

cells.append(nbf.v4.new_markdown_cell("## 2. Cloner les repos"))
cells.append(nbf.v4.new_code_cell(
"""%cd /content
!git clone https://github.com/bono-p/fuv-tts-vits.git
!git clone https://github.com/ylacombe/finetune-hf-vits.git"""
))

cells.append(nbf.v4.new_markdown_cell("## 3. Installer les dépendances"))
cells.append(nbf.v4.new_code_cell(
"""%cd /content/fuv-tts-vits
!pip install -r requirements.txt --quiet

%cd /content/finetune-hf-vits
!pip install -r requirements.txt --quiet

# Construction de l'extension Cython pour l'alignement monotone (obligatoire,
# la version Python pure est beaucoup trop lente)
%cd monotonic_align
!mkdir -p monotonic_align
!python setup.py build_ext --inplace
%cd .."""
))

cells.append(nbf.v4.new_markdown_cell(
"## 4. Connexion au Hugging Face Hub\\n\\n"
"Nécessaire pour publier le dataset assemblé et le modèle fine-tuné. "
"Génère un token avec les droits 'write' sur "
"https://huggingface.co/settings/tokens"
))
cells.append(nbf.v4.new_code_cell(
"""from huggingface_hub import notebook_login
notebook_login()"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 5. Nettoyer l'audio (silence + normalisation)\\n\\n"
"Ne touche pas aux fichiers originaux du dataset STT : écrit dans un "
"dossier séparé `audio_tts_clean/` + `metadata_tts.json`."
))
cells.append(nbf.v4.new_code_cell(
"""%cd /content/fuv-tts-vits/src
!python clean_audio.py"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 6. Assembler et publier le dataset sur le Hub\\n\\n"
"Publié en **privé** par défaut (voir `hub_config.dataset_repo_id` dans "
"`src/config.py` si tu veux changer le nom du repo)."
))
cells.append(nbf.v4.new_code_cell(
"""!python prepare_dataset.py"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 7. Convertir le checkpoint MMS-TTS-ful (une seule fois)\\n\\n"
"Ajoute le discriminateur nécessaire à l'entraînement (absent du "
"checkpoint d'inférence standard). À ne faire qu'une fois — le résultat "
"est réutilisable pour de futurs entraînements."
))
cells.append(nbf.v4.new_code_cell(
"""%cd /content/finetune-hf-vits
!python convert_original_discriminator_checkpoint.py \\\\
    --language_code ful \\\\
    --pytorch_dump_folder_path /content/mms-tts-ful-with-discriminator \\\\
    --push_to_hub bonopassale/mms-tts-ful-with-discriminator"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 8. Lancer le fine-tuning\\n\\n"
"Le fichier `training_config/finetune_fuv.json` (dans `fuv-tts-vits`) "
"pointe déjà vers le dataset et le checkpoint convertis ci-dessus. "
"Ajuste `num_train_epochs`, `learning_rate` ou les poids de loss "
"directement dans ce fichier si besoin, avant de lancer.\\n\\n"
"Avec un petit dataset (quelques centaines de clips courts), compte "
"20 à 60 minutes sur un GPU T4."
))
cells.append(nbf.v4.new_code_cell(
"""!accelerate launch /content/finetune-hf-vits/run_vits_finetuning.py \\\\
    /content/fuv-tts-vits/training_config/finetune_fuv.json"""
))

cells.append(nbf.v4.new_markdown_cell("## 9. Tester le modèle fine-tuné"))
cells.append(nbf.v4.new_code_cell(
"""from transformers import pipeline
import scipy

model_id = "bonopassale/tts-fuv-voice"  # ou le chemin local output_dir
synthesiser = pipeline("text-to-speech", model_id, device=0)

speech = synthesiser("Jam waali")

scipy.io.wavfile.write("test_tts_fuv.wav", rate=speech["sampling_rate"], data=speech["audio"][0])

from IPython.display import Audio
Audio("test_tts_fuv.wav")"""
))

nb['cells'] = cells

with open('/home/claude/fuv-tts-vits/notebook/finetune_tts_fuv.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook TTS cree avec succes.")
