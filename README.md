# fuv-tts-vits

Fine-tuning de **MMS-TTS (architecture VITS)** pour la synthèse vocale
(TTS) en **fulfulde**, sur une **voix unique** (la tienne), en réutilisant
le dataset audio déjà constitué pour le projet STT (`fuv-stt-whisper`).

---

## 1. Pourquoi ce modèle et cet outil de fine-tuning ?

- **`facebook/mms-tts-ful`** : checkpoint VITS pré-entraîné sur le fulfulde
  (macrolangue "ful"), avec un tokenizer **caractère** (pas besoin de
  phonémiseur externe type espeak-ng, contrairement au VITS anglais
  original) — adapté aux langues sans ressources G2P constituées.
- **`ylacombe/finetune-hf-vits`** : outil de fine-tuning officiel HF pour
  VITS/MMS, avec une recette qui donne de bons résultats en **20 minutes**
  avec seulement **80 à 150 échantillons**. Notre dataset (~2h, quelques
  centaines de clips) est largement suffisant.

**Rappel important** (déjà discuté pour le STT) : `mms-tts-ful` est
entraîné sur le code macrolangue "ful", pas garanti être exactement la
variété Adamaoua/Cameroun. Comme on fine-tune entièrement sur ta propre
voix et ton propre texte, ça reste un bon point de départ — le fine-tuning
réapprend ta prononciation par-dessus, quelle que soit la variété
d'origine du checkpoint.

---

## 2. Comment fonctionne VITS (rappel)

VITS est un auto-encodeur variationnel conditionnel entraîné de bout en
bout texte → forme d'onde :
- **Encodeur de texte** (Transformer) → représentation contextuelle des tokens
- **Prédicteur de durée stochastique** → gère le rythme, aligne texte/audio
  sans annotation manuelle
- **Flux normalisant (prior)** → enrichit la distribution latente conditionnée sur le texte
- **Encodeur a posteriori** (entraînement seulement) → encode l'audio réel
- **Décodeur** → génère la forme d'onde brute directement (style HiFi-GAN)
- **Discriminateur adversarial (GAN)** → pousse le résultat à sonner plus naturel

Le fine-tuning ajoute un discriminateur au checkpoint d'inférence standard
(absent par défaut), d'où l'étape de conversion à l'étape 7 ci-dessous.

---

## 3. Différences avec le dataset STT — étape de nettoyage

Le dataset STT (multi-voix à l'origine, maintenant tout sur ta voix) n'est
pas directement optimal pour le TTS : le TTS est plus sensible au bruit de
fond, aux silences mal coupés, et aux variations de volume qu'un modèle
STT (qui apprend à ignorer ce genre de bruit).

`src/clean_audio.py` corrige ça **sans toucher aux fichiers originaux** :
- coupe le silence en trop en début/fin de chaque clip (seuil réglable
  via `top_db` dans `config.py`)
- rajoute une petite marge de silence propre (`silence_padding_sec`)
- normalise le volume à un niveau de crête constant (`target_peak_dbfs`)
- écrit tout dans un dossier séparé `audio_tts_clean/` + un
  `metadata_tts.json` dédié

Si tu vois trop (ou pas assez) de silence coupé sur les résultats, ajuste
juste `top_db` dans `config.py` (plus haut = coupe plus agressivement) et
relance le script.

---

## 4. Structure du projet

```
fuv-tts-vits/
├── README.md
├── requirements.txt
├── build_notebook.py
├── src/
│   ├── config.py            # tous les chemins et identifiants Hub
│   ├── clean_audio.py         # nettoyage audio (silence + volume)
│   └── prepare_dataset.py     # assemble + publie le dataset sur le Hub
├── training_config/
│   └── finetune_fuv.json      # config pour run_vits_finetuning.py
└── notebook/
    └── finetune_tts_fuv.ipynb # notebook Colab complet
```

---

## 5. Utilisation (Google Colab)

Ouvre `notebook/finetune_tts_fuv.ipynb` sur Colab (GPU T4 activé) et
exécute les cellules dans l'ordre :

1. **Montage Drive** — ton dataset `fuv-stt-dataset/` doit déjà y être.
2. **Clone des repos** — ce repo (`fuv-tts-vits`) + l'outil d'entraînement
   officiel `ylacombe/finetune-hf-vits`.
3. **Installation des dépendances** des deux repos + compilation de
   l'extension Cython pour l'alignement monotone (obligatoire — la
   version Python pure est bien trop lente).
4. **Connexion au Hugging Face Hub** (`notebook_login()`) — nécessaire
   car le dataset assemblé et le modèle fine-tuné sont poussés sur le Hub.
5. **Nettoyage audio** (`clean_audio.py`).
6. **Publication du dataset** (`prepare_dataset.py`) — en **privé** par
   défaut, vers `bonopassale/fuv-tts-monospeaker` (modifiable dans
   `src/config.py`).
7. **Conversion du checkpoint** `mms-tts-ful` (ajout du discriminateur)
   — à faire **une seule fois**, réutilisable pour de futurs entraînements.
8. **Lancement du fine-tuning**, piloté par `training_config/finetune_fuv.json`.
9. **Test du modèle** obtenu, directement via le pipeline `text-to-speech`.

---

## 6. Ajuster l'entraînement

Tout se règle dans `training_config/finetune_fuv.json` :

| Paramètre | Rôle | Valeur par défaut |
|---|---|---|
| `num_train_epochs` | Nombre de passages complets sur le dataset | 200 |
| `learning_rate` | Taux d'apprentissage | 2e-5 |
| `per_device_train_batch_size` | Taille de batch | 16 |
| `weight_mel` / `weight_kl` / `weight_disc` / etc. | Pondération des différentes pertes (qualité spectrale vs réalisme adversarial vs latent) | valeurs par défaut de la recette officielle |

Avec un dataset de quelques centaines de clips courts, 200 epochs tourne
généralement en 20 à 60 minutes sur un T4. Si la voix de sortie sonne
"instable" ou trop robotique, les premiers leviers à essayer sont
`weight_mel` (qualité spectrale) et `num_train_epochs`.

---

## 7. Tester le modèle après entraînement

```python
from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-speech", "bonopassale/tts-fuv-voice", device=0)
speech = synthesiser("Jam waali")
scipy.io.wavfile.write("sortie.wav", rate=speech["sampling_rate"], data=speech["audio"][0])
```

---

## 8. Limites connues

- Un dataset ~2h mono-locuteur donne un **premier prototype convaincant**,
  pas un modèle de production. La stabilité prosodique s'améliore avec
  plus de données et plus de variété de contextes/phrases.
- Le VITS/MMS d'origine étant non-déterministe (durée stochastique), deux
  générations du même texte peuvent différer légèrement en rythme —
  normal, fixe une seed si tu veux de la reproductibilité stricte.
- Si des mots/tournures très spécifiques au fulfulde restent mal
  prononcés, la qualité s'améliore surtout en enrichissant le corpus
  texte avec plus d'exemples de ces cas précis, plutôt qu'en modifiant
  les hyperparamètres.
