from huggingface_hub import login
from pdf2text import pdf_to_text
from model_service import ModelService
from requierement import Requirement

# ----------------------------------------------------------
# 1. (Optionnel) Login HuggingFace si ton modèle est privé
# ----------------------------------------------------------
# Si le modèle est public, tu peux supprimer cette ligne
login(token="hf_xxx")


# ----------------------------------------------------------
# 2. Charger le dernier modèle performant de Mistral
# ----------------------------------------------------------
models_to_load = ["static/models/Llama-3.2-3B-Instruct"]

service = ModelService(models_to_load=models_to_load)


# ----------------------------------------------------------
# 3. Convertir ton PDF → texte
# ----------------------------------------------------------
pdf_path = "./test.pdf"
document_text = pdf_to_text(pdf_path)

print("\n📄 PDF chargé, longueur du texte :", len(document_text), "caractères\n")


# ----------------------------------------------------------
# 4. Appeler l’extraction d’exigences
# ----------------------------------------------------------
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

requirements: list[Requirement] = service.generate_requirements_from_document(
    model_id=model_id,
    document=document_text
)


# ----------------------------------------------------------
# 5. Affichage formaté
# ----------------------------------------------------------
print("\n==============================")
print("        RÉSULTATS")
print("==============================")

for i, r in enumerate(requirements, 1):
    print(f"\n--- Exigence {i} ---")
    print("Principale :", r.exigence_principale)
    print("Sous-exigences :", r.sous_exigences)
    print("Implicites :", r.implicites)
    print("Description :", r.description)
