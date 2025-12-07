import os
import shutil
import zipfile
from glob import glob
import json
from pathlib import Path

# --- Caches HF dans /data ---
os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/hf/transformers")
os.environ.setdefault("HF_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/hf/datasets")

from huggingface_hub import hf_hub_download
from PIL import Image

import torch
import torch.distributed as dist
from transformers import AutoProcessor, AutoModelForVision2Seq

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN manquant")

REPO_ID = "jnWhisper/vlog_ai"
ZIP_FILENAME = "vlog_ai.zip"

DATA_DIR = "/data"
OUT_DIR = "/output"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("PALIGEMMA_MODEL_ID", "google/paligemma2-10b-ft-docci-448")
PROMPT = "<image>caption en"
FINAL_OUTPUT_FILE = os.path.join(OUT_DIR, "captions_finetune.jsonl")


def dist_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def barrier(dist_on: bool):
    if dist_on:
        dist.barrier()


def download_and_extract_once(rank: int):
    if rank != 0:
        return

    print("📥 Téléchargement du dataset (ZIP) depuis Hugging Face...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ZIP_FILENAME,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=DATA_DIR,
        local_dir_use_symlinks=False,
    )

    print("🗂 Décompression du ZIP...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    imgs = []
    for ext in exts:
        imgs.extend(glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
    imgs = sorted(imgs)

    with open(os.path.join(DATA_DIR, "images_list.txt"), "w", encoding="utf-8") as f:
        for p in imgs:
            f.write(p + "\n")

    print(f"📸 {len(imgs)} images listées.")


def load_model(local_rank: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible dans le conteneur.")

    device = f"cuda:{local_rank}"

    # A100 -> bfloat16 recommandé ; sinon fp16
    major, _minor = torch.cuda.get_device_capability(local_rank)
    dtype = torch.bfloat16 if major >= 8 else torch.float16

    print(f"🧠 Chargement modèle {MODEL_ID} sur {device} (dtype={dtype}) ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    ).eval()

    print("✅ Modèle chargé.")
    return processor, model, device


def caption_one(path, processor, model, device):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, text=PROMPT, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=220, do_sample=False)

    prompt_len = inputs["input_ids"].shape[-1]
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def main():
    dist_on, rank, world_size, local_rank = dist_init()

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("device_count =", torch.cuda.device_count())
    print(f"rank={rank} world_size={world_size} local_rank={local_rank}")

    download_and_extract_once(rank)
    barrier(dist_on)

    list_path = os.path.join(DATA_DIR, "images_list.txt")
    if not os.path.exists(list_path):
        raise RuntimeError("images_list.txt manquant")

    with open(list_path, "r", encoding="utf-8") as f:
        all_images = [l.strip() for l in f if l.strip()]

    my_images = all_images[rank::world_size]
    partial_path = os.path.join(OUT_DIR, f"captions_rank{rank}.jsonl")

    processor, model, device = load_model(local_rank)

    print(f"🚀 rank {rank}: {len(my_images)} images à traiter")
    with open(partial_path, "w", encoding="utf-8") as out_f:
        for i, p in enumerate(my_images, 1):
            try:
                cap = caption_one(p, processor, model, device)
                out_f.write(json.dumps({"image": os.path.basename(p), "text": cap}, ensure_ascii=False) + "\n")
                if i % 25 == 0:
                    print(f"rank {rank}: {i}/{len(my_images)}")
            except Exception as e:
                print(f"rank {rank} erreur {p}: {e}")

    barrier(dist_on)

    if rank == 0:
        print("🧩 Merge des fichiers JSONL...")
        with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as final_f:
            for r in range(world_size):
                p = os.path.join(OUT_DIR, f"captions_rank{r}.jsonl")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as in_f:
                        shutil.copyfileobj(in_f, final_f)

        print("✅ JSONL final:", FINAL_OUTPUT_FILE)
        print("📦 Taille JSONL final (bytes):", os.path.getsize(FINAL_OUTPUT_FILE))

    barrier(dist_on)
    if dist_on:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
