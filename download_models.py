from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download

# 1. Téléchargement complet (cache ou dossier local)
repo_id = "google/flan-t5-base"
local_dir = "./static/models/flan-t5-base"
# repo_id = "sentence-transformers/all-MiniLM-L6-v2"
# local_dir = "./static/models/all-MiniLM-L6-v2"
# for RAG system : entence-transformers/all-mpnet-base-v2
# little ressources: flan-t5-base, google/flan-t5-small,  google/long-t5-tglobal-base, allenai/led-base-16384
# orca-mini:3b-q4_0

# google/flan-t5-large
# google/flan-t5-small
# google/flan-t5-base
# google/flan-t5-base

# instructGPT

# Option B — charger via transformers & sauvegarder localement
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, low_cpu_mem_usage=True)  # ou dtype approprié

tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)
