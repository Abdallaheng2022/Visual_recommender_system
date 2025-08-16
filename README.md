# Generative vs. Vector vs. RAG — for a 44K‑item Visual Recommender

This README explains three approaches you’ll see in this repo — **Generative (LLM‑only), Vector Search, and RAG (Retrieval‑Augmented Generation)** — and maps each to the included Streamlit apps/scripts. It’s tailored for the **Fashion Product Images (Small)** dataset (~44k items).

[![Demo Video](https://drive.google.com/thumbnail?id=1QYWSd6N0mjRKySpYhcnP96jO9t1Ai5rh&sz=w640-h360)](https://drive.google.com/file/d/1QYWSd6N0mjRKySpYhcnP96jO9t1Ai5rh/view?usp=sharing)


---

## 1) Plain Generative (LLM‑only)

**What it is**
- You give the LLM your *query* and a **small list of candidate products** (JSON with ids + descriptions).  
- The LLM **generates** a ranked list with scores + justifications.
- No external retrieval step; the model only “sees” what you pass in.

**Pros**
- Easiest to prototype; rich natural‑language justifications.
- Works even without a vector DB.

**Cons**
- Can miss relevant items not included in the candidate list.
- Sensitive to prompt/format; can “hallucinate” ids if not constrained.
- Token limits cap how many products you can pass.

**Typical flow**
1. (Optional) **Rephrase user intent** to normalize the query.
2. Pass *subset* of stock items + intent to LLM.
3. LLM returns `[{"id", "score", "justification"}...]`.

**Files in repo**
- `text-based_query_visual_recommender_system.py`  
  Minimal generative recommender using a list of stock items and a user query.
- `Text_based_with_intent_rephraser.py`  
  Same idea but includes a **query rephraser** step before LLM ranking.

---

## 2) Vector Search (pure retrieval)

**What it is**
- Convert product **descriptions** (and optionally titles/tags) into **embeddings**.
- Store them in a **vector index** (FAISS here).
- Retrieve top‑k nearest neighbors to the user query embedding.

**Pros**
- Scales to **44k+** products with fast k‑NN.
- Deterministic, explainable recall; no prompt cost to retrieve.
- Great for “more‑like‑this” and semantic search.

**Cons**
- Retrieval only — no rich natural‑language justification out of the box.
- Quality depends on embedding model + text quality.

**Typical flow**
1. Build FAISS once from product descriptions.
2. At query time, embed the query and **retrieve top‑k** products.
3. Show results (optionally add scores, facets).

**Files in repo**
- `Vector_based_Visual_recommender.py`  
  Builds FAISS, retrieves top‑k with `similarity_search[_with_score]`, and displays results.

---

## 3) RAG (Retrieval‑Augmented Generation)

**What it is**
- **Retrieve** candidates with vectors **first**, then **generate** the final ranking/summary with the LLM using the retrieved items as context.
- Best of both: high‑recall retrieval + LLM explanations/reranking.

**Pros**
- More factual/grounded than plain LLM; less hallucination.
- Handles large catalogs (retrieve 50–200, then LLM reranks top‑N).

**Cons**
- Requires both vector infra and prompt design.
- Two‑stage latency (retrieve → rerank).

**Typical flow**
1. (Optional) Rephrase/normalize the user query.
2. Vector **retrieve** top‑k product candidates.
3. Pack candidates (ids + short descriptions) into a compact JSON prompt.
4. LLM **reranks** and explains: returns exact ids + scores + justifications.

**Files in repo**
- `RAG_Text_based_query_visual_recommender.py`  
  Text‑only RAG: FAISS shortlist → LLM rerank & justify.
- `RAG_image_with_text_based_query.py`  
  **Multimodal RAG**: can fold liked images or an image description (via vision model) into the query, retrieve, then rerank with LLM.
- `image_with_text_based_query_RAG_VS_Recommender.py` / `Fashion-items_DALE.py`  
  Demo UIs that compare **RAG** vs **Generative** flows, plus “like” interactions to bias the query.

---

## Dataset note (44k items)

These apps assume the **Fashion Product Images (Small)** dataset (≈44,000 products with images + labels). You’ll see code reading a CSV like:

```
/.../fashion_product_small/styles{nrows}_with_decription.csv
```

Index this CSV text (and optionally derive extra fields) into your vector store. Use the image folder to display product thumbnails in Streamlit.

---

## When should I use which?

| Goal | Choose | Why |
| --- | --- | --- |
| Fast semantic search over all 44k | **Vector** | Scalable recall; zero prompt cost per candidate. |
| Trustworthy top‑N with clear explanations | **RAG** | Grounded by retrieval; LLM explains/reranks. |
| Quick demo without vector infra | **Generative** | Minimal setup; good for POCs with small candidate lists. |

---

## Practical tips

- **Cap the rerank set**: retrieve 50–200 with vectors; ask the LLM for top‑10/20.  
- **Constrain IDs**: tell the LLM to only return ids from the provided list.  
- **Keep JSON compact**: id + short description to control tokens.  
- **Cache FAISS**: build once per dataset load.  
- **Show provenance**: display the product image and key attributes for trust.

---

## Quick map from UI features to code

- **Image grid + likes**: grid components with stable widget keys in `*image*`/`*DALE*.py` demos.  
- **Intent rephraser**: turns noisy text into a clean shopping intent before retrieval/rerank.  
- **Vision add‑on**: describe a liked/uploaded image, append to the text query for better matches.  
- **Rerank prompt**: JSON with `{"id","description"}` candidates and a request to return `top n` with `score` and `justification`.

---

## Run locally

1. Put your `.env` with the relevant keys (OpenAI, Azure if used).  
2. Ensure the CSV + images are in the paths referenced by the scripts (or edit paths).  
3. Install deps:
   ```bash
   pip install streamlit pandas pillow faiss-cpu langchain-community langchain-openai openai tqdm matplotlib python-dotenv
   ```
4. Launch any app, e.g.:
   ```bash
   streamlit run RAG_Text_based_query_visual_recommender.py
   ```

---

## References (general reading)

- Retrieval‑Augmented Generation (original paper & surveys)  
  - Lewis et al., 2020 (NeurIPS).  
  - Recent RAG surveys/tutorials for production patterns.

- Vector search & similarity
  - Cosine similarity and vector metrics (cosine/dot/Euclidean).  
  - Practical guides from vector‑DB vendors.

- Dataset
  - “Fashion Product Images (Small)” (~44k items).

*(This README intentionally keeps references high‑level; see the project notes or your documentation site for full links.)*
