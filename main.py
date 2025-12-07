# main.py
import os
import shutil
import zipfile
from glob import glob
from huggingface_hub import hf_hub_download

# 1) Récupérer le token HF depuis l'environnement
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Erreur : la variable d'environnement HF_TOKEN n'est pas définie.")

# 2) Infos sur ton dataset HF
REPO_ID = "jnWhisper/vlog_ai"
ZIP_FILENAME = "vlog_ai.zip"   # nom EXACT du zip sur Hugging Face
DATA_DIR = "/data"

os.makedirs(DATA_DIR, exist_ok=True)

def download_zip():
    print("📥 Téléchargement du fichier depuis Hugging Face...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ZIP_FILENAME,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    dest_path = os.path.join(DATA_DIR, ZIP_FILENAME)
    shutil.copy(zip_path, dest_path)
    print(f"✅ ZIP copié vers : {dest_path}")
    return dest_path

def extract_zip(zip_path):
    print("🗂 Décompression du ZIP...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print("✅ Décompression terminée.")

    # Lister les images trouvées
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
    print(f"📸 {len(image_paths)} images trouvées dans {DATA_DIR}")
    return image_paths

def main():
    zip_path = download_zip()
    _ = extract_zip(zip_path)
    print("✅ Étape 1 (download + unzip) terminée. PaliGemma viendra ensuite 😉")

if __name__ == "__main__":
    main()
