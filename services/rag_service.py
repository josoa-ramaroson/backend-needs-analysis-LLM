import re
import math
from typing import List, Dict, Tuple
from tqdm import tqdm

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from model_service import ModelService
from llama_model_service import LlamaModelService
import logging
logger = logging.getLogger(__name__)


class RAGService(ModelService):
    
    def __init__(
                    self, 
                    llm_service,
                    #embed_model_name="sentence-transformers/all-mpnet-base-v2",
                    #embed_model_name="intfloat/e5-large",
                    #embed_model_name="nomic-ai/nomic-embed-text-v1.5",
                    embed_model_name="Qwen/Qwen3-Embedding-0.6B",
                    crossencoder_name="mixedbread-ai/mxbai-rerank-base-v2"
                    #crossencoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
                    #crossencoder_name="cross-encoder/stsb-roberta-large"
                    #crossencoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                    ):
        self.llm_service = llm_service
        self.embed_model = SentenceTransformer(embed_model_name, trust_remote_code=True)
        self.cross_encoder = CrossEncoder(crossencoder_name)
    
    def simple_chunk_text(self, text: str, max_words: int = 200, overlap_words: int = 50) -> List[Dict]:
        """
        Coupe le texte en chunks basés sur les mots (approx pour tokens).
        Retourne une liste de dicts: {'chunk_id': int, 'text': str, 'words': int}
        """
        # split into paragraphs or sentences as base units
        paragraphs = re.split(r'\n{2,}|\r\n{2,}', text)
        chunks = []
        chunk_words = []
        cur = []
        cur_len = 0
        chunk_id = 0
        for p in paragraphs:
            # normalize whitespace
            p = p.strip()
            if not p:
                continue
            words = p.split()
            wlen = len(words)
            # if paragraph alone exceeds max, split it
            if wlen >= max_words:
                # split paragraph into windows of max_words with overlap
                i = 0
                while i < wlen:
                    part = words[i:i+max_words]
                    chunks.append({"chunk_id": chunk_id, "text": " ".join(part), "words": len(part)})
                    chunk_id += 1
                    i += max_words - overlap_words
                continue
            # otherwise accumulate paragraphs until reach max
            if cur_len + wlen <= max_words:
                cur.append(p)
                cur_len += wlen
            else:
                # flush current chunk
                chunks.append({"chunk_id": chunk_id, "text": " ".join(cur), "words": cur_len})
                chunk_id += 1
                # start new chunk with this paragraph
                cur = [p]
                cur_len = wlen
        # final flush
        if cur:
            chunks.append({"chunk_id": chunk_id, "text": " ".join(cur), "words": cur_len})
        return chunks
    
    def build_bm25(self, chunks: List[Dict]):
        tokenized = [re.findall(r"\w+", c['text'].lower()) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        return bm25, tokenized

    def bm25_retrieve(self, bm25, tokenized_chunks, query: str, top_n: int = 50) -> List[Tuple[int, float]]:
        q_tokens = re.findall(r"\w+", query.lower())
        scores = bm25.get_scores(q_tokens)  # numpy array
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_idx]
    def dense_rerank(self, 
                    query: str,
                    candidates: List[Dict],
                    batch_size: int = 32) -> List[Tuple[int, float]]:
        # compute query embedding
        q_vec = self.embed_model.encode([query], convert_to_numpy=True)
        # compute candidate embeddings in batch
        texts = [c['text'] for c in candidates]
        cand_vecs = self.embed_model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
        # cosine similarity
        sims = cosine_similarity(q_vec, cand_vecs)[0]  # shape (n_candidates,)
        idx_and_scores = [(candidates[i]['chunk_id'], float(sims[i])) for i in range(len(candidates))]
        idx_and_scores.sort(key=lambda x: x[1], reverse=True)
        return idx_and_scores
    
    def cross_encoder_rerank(self, 
                            query: str,
                            candidates: List[Dict],
                            top_m: int = 20,
                            batch_size: int = 32) -> List[Tuple[int, float]]:
        # Build pairs for cross-encoder: (query, passage)
        pairs = [(query, c['text']) for c in candidates[:top_m]]
        scores = self.cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        results = [(candidates[i]['chunk_id'], float(scores[i])) for i in range(len(scores))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def retrieve_top_passages(self, 
                            text: str,
                            query: str,
                            bm25_top_n: int = 80,
                            dense_top_k: int = 20,
                            final_top_k: int = 10):
        # 1. chunk
        print("Chunk preparation")
        chunks = self.simple_chunk_text(text, max_words=400, overlap_words=80)
        # 2. BM25
        print("BM25 preparation")
        bm25, tokenized = self.build_bm25(chunks)
        bm25_results = self.bm25_retrieve(bm25, tokenized, query, top_n=bm25_top_n)
        # select candidate dicts
        candidates = []
        idx_to_chunk = {c['chunk_id']: c for c in chunks}
        for idx, score in bm25_results:
            c = idx_to_chunk[idx]
            c_copy = c.copy()
            c_copy['bm25_score'] = score
            candidates.append(c_copy)
        # 3. Embeddings rerank (dense) - compute embeddings for the candidates only
        bm25_score_map = {c['chunk_id']: c['bm25_score'] for c in candidates}
        print("rank embedding")
        dense_ranked = self.dense_rerank(query, candidates)
        # keep top dense_top_k chunk_ids in order
        dense_top_ids = [cid for cid, s in dense_ranked[:dense_top_k]]
        dense_candidates = [idx_to_chunk[cid] for cid in dense_top_ids]
        # 4. Cross-encoder rerank on the dense_candidates
        
        cross_ranked = self.cross_encoder_rerank( query, dense_candidates, top_m=len(dense_candidates))
        # final top K
        final_ids = [cid for cid, s in cross_ranked[:final_top_k]]
        final_passages = [idx_to_chunk[cid] for cid in final_ids]
        # attach scores
        # find corresponding dense score and cross score for provenance
        dense_score_map = dict(dense_ranked)
        cross_score_map = dict(cross_ranked)
        results = []
        for cid in final_ids:
            results.append({
                "chunk_id": cid,
                "text": idx_to_chunk[cid]['text'],
                "bm25_score": bm25_score_map.get(cid),
                "dense_score": dense_score_map.get(cid),
                "cross_score": cross_score_map.get(cid)
            })
        return results

    def merge_passages(self, passages_results):
        parts = []

        for r in passages_results:
            # safe access and formatting of scores (they may be None)
            cross = r.get("cross_score")
            dense = r.get("dense_score")
            bm25  = r.get("bm25_score")

            # format scores nicely (or "N/A" if missing)
            cross_s = f"{cross:.4f}" if cross is not None else "N/A"
            dense_s = f"{dense:.4f}" if dense is not None else "N/A"
            bm25_s  = f"{bm25:.4f}" if bm25 is not None else "N/A"

            # header with provenance
            header = f"CHUNK {r['chunk_id']} | CROSS {cross_s} | DENSE {dense_s} | BM25 {bm25_s}"

            text = r['text'].strip()
            
            # build the chunk block and append
            parts.append(header + "\n" + text)

        # join all chunk blocks into one document string (with a clear separator)
        document = "\n".join(parts)
        return document

    def build_prompt(self, document):
        return self.PROMPT_TEMPLATE.format(document=document)
    def extract(self, document: str):
        query = "give all explicit and implicit requirements and constraints for this project, functional requirements, non-functional requirements (performance, security, reliability, scalability)"
        passages_results = self.retrieve_top_passages(document, query,
                                        bm25_top_n=70,
                                        dense_top_k=20,
                                        final_top_k=10)

        merged_passages = self.merge_passages(passages_results)
        print("Prompting...")
        prompt = self.build_prompt(merged_passages)
        llm_response = self.llm_service._call(prompt)
        print(llm_response)
        return llm_response

#testing
text_doc = """
Gianah (Analyste des besoins) :
“Hello everyone! My name is Gianah, and I’m the Business Analyst for this project. I will be guiding
our meeting today.
We are here because we want to build a mobile app for the university library. This app will help
students and staff use the library services in a better way.
Before we start talking about the app, I want to introduce you to the team. Each person here has a
special role to play in this project. Let’s meet them now.”
Manampy (Étudiant):
“Hi, my name is Manampy.
I am here to represent the students who will use this app every day. I will
share what students need and expect from the app.”
Benjamina (Bibliothécaire):
“Hello, I’m Ms. Benjamina. I am the head librarian.
My job is to manage the lending and returning of books. I make sure everything works well for both the
library staff and the students.”
Rado (Responsable pédagogique) :
“Hello, I’m Mr. Rado.
My role is to make sure the app meets the needs of teachers and students when
it comes to their learning and teaching.”
Nambinintsoa (Responsable informatique ):
“Hello, my name is Nambinintsoa.
I am the IT manager.
I will make sure the app works well with our current computer systems. I will also make sure it is safe
and easy to use.”
Gianah (Analyste des besoins) :
“Thank you all for the introductions.Now that we know who everyone is and their role, let’s start our discussion about the app.
Our goal is to understand what each of you expects from the app. We want to know what you need, and
what problems or limits we should keep in mind.
Manampy, let’s begin with you. What do you think students need most from this app?”
Manampy (Étudiant):
“Thank you, Gianah. For students, the most important thing is to be able to search the library catalog
from our phones.
This means we can see if books are available without going to the library. That saves time.
It would also be very helpful to be able to reserve books directly from the app, especially during exams
when we are very busy.
I would like the app to send me notifications before the due date so I don’t forget to return the books on
time.
Lastly, it would be great if I can extend the loan for a book using the app, so I don’t have to go to the
library just to ask for more time.”
Benjamina (Bibliothécaire):
“For us in the library, the biggest problem is answering many questions about whether a book is
available.
If students can check this on their own using the app, it will help us a lot.
We also want the reservation process to be automatic, so it is easier to manage the loans.
A QR code scanner in the app would help us quickly record when books are borrowed or returned. This
will reduce mistakes.
But we need to make sure that students can only extend a loan if nobody else has reserved the book
after them.”
Rado (Responsable pédagogique) :
“I think this app can help with learning, too.
It would be good if teachers could suggest books related to their courses inside the app.This way, students will know which books are important for their classes.
We could also have lists of books by subjects or by semester to make it easier to find what you need.”
Nambinintsoa (Responsable informatique ):
“From a technical side, the app will need to connect to our library system called KohaTek. It will use
something called an API to get live information about books and users.
Security is very important. Students and staff will log in using their university usernames and
passwords.
To reach many people, it is best to build a hybrid app. That means it will work on both Android phones
and iPhones.
We will also add notifications to remind users about due dates, and make sure the app works well with
QR code scanning inside the library.”
Gianah (Analyste des besoins) :
“Thank you all for sharing your ideas and needs.
Let me quickly summarize the main points we talked about today:**
• Students want to search and browse the catalog on their phones.
• They want to reserve books and extend loans through the app.
• The app should send reminders to avoid late returns.
• QR code scanning will make loans and returns faster.
• Teachers can suggest books related to their courses.
• The app must have secure login with university accounts.
• The app will work on both Android and iOS and connect to KohaTek.
“Our next step is to write down these needs clearly and start designing the first version of the app.
Thank you all for your time and help. Working together will make this project a success!”
"""
llama_model_service = LlamaModelService()
rag_service = RAGService(llama_model_service)
result = rag_service.getRequierementAsJson(text_doc)
print(result)