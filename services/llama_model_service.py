# services/llama_model_service.py
from typing import Any, Optional
from services.model_service import ModelService, ModelServiceError
import logging

logger = logging.getLogger(__name__)


class LlamaModelService(ModelService):
    """
    Implémentation de ModelService pour un modèle local Llama (pipeline transformers).
    Par défaut, utilise le chemin ./static/models/Llama-3.2-3B-Instruct si aucun model_id donné.
    """

    def __init__(self, model_id: str = "./static/models/Llama-3.2-3B-Instruct", device: str = "cpu"):
        """
        :param model_id: chemin local ou identifiant HF du modèle (ici par défaut le dossier local)
        :param device: 'cpu' ou 'cuda' (ou indice GPU comme 'cuda:0' si tu veux)
        """
        super().__init__(model_id=model_id, device=device)
        self.pipe: Optional[Any] = None

    def load(self) -> None:
        """
        Charge la pipeline transformers depuis le dossier local (ou repo id).
        Configure le paramètre `device` pour la pipeline :
         - device = -1 => CPU
         - device = 0  => premier GPU (si disponible)
        """
        try:
            from transformers import pipeline
        except Exception as e:
            raise ModelServiceError(f"Transformers non disponible : {e}")

        # Déterminer la cible d'exécution pour pipeline : -1 pour CPU, 0 pour GPU par défaut
        if isinstance(self.device, str) and self.device.lower().startswith("cuda"):
            device_arg = 0
        elif isinstance(self.device, str) and self.device.isdigit():
            # support simple d'indice fourni en string
            device_arg = int(self.device)
        elif isinstance(self.device, int):
            device_arg = self.device
        else:
            device_arg = -1
        
        try:
            # charge le modèle & tokenizer à partir du chemin local (ou HF id)
            # trust_remote_code=False pour sécurité ; change si nécessaire
            self.pipe = pipeline(
                "text-generation",
                model=self.model_id,
                tokenizer=self.model_id,
                device=device_arg,
                trust_remote_code=True,
            )
            self._is_loaded = True
            logger.info("LlamaModelService loaded model '%s' on device=%s", self.model_id, self.device)
        except Exception as e:
            raise ModelServiceError(f"Échec du chargement du modèle '{self.model_id}': {e}")

    def unload(self) -> None:
        """
        Décharge la pipeline et tente de libérer la mémoire GPU si disponible.
        """
        try:
            # supprimer la reference à la pipeline
            if self.pipe is not None:
                try:
                    # pipeline peut contenir des objets torch -> tenter nettoyage
                    import torch  # type: ignore
                    # supprimer références
                    del self.pipe
                    self.pipe = None
                    # vider cache GPU si torch installé
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception:
                    # si torch introuvable ou suppression différente, on supprime quand même la pipe
                    self.pipe = None
        finally:
            self._is_loaded = False
            logger.info("LlamaModelService unloaded model '%s'", self.model_id)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, do_sample: bool = False, **kwargs) -> str:
        """
        Génère du texte via la pipeline transformers.
        Retourne la sortie brute (string) renvoyée par le pipeline.
        """
        if not self._is_loaded or self.pipe is None:
            raise ModelServiceError("Model not loaded. Call load() before generate().")

        try:
            # paramètres par défaut : deterministic quand temperature==0 et do_sample False
            gen = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                return_full_text=False,
                **kwargs,
            )

            # Pipeline retourne généralement une liste de dicts
            if isinstance(gen, list) and len(gen) > 0 and "generated_text" in gen[0]:
                raw = gen[0]["generated_text"]
            elif isinstance(gen, dict) and "generated_text" in gen:
                raw = gen["generated_text"]
            else:
                # fallback : tenter str()
                raw = str(gen)

            return raw
        except Exception as e:
            raise ModelServiceError(f"Échec de génération: {e}")
