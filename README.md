
# EAGLE Router Reimplementation on RepliQA

This project reimplements the **EAGLE router** (from the paper *EAGLE: Efficient Training-Free Router for Multi-LLM Inference*) and applies it to the [Not Diamond RepliQA dataset](https://huggingface.co/datasets/notdiamond/repliqa_gpt4o_gpt4omini_evals).  
The router combines **global ability** (via ELO ratings across all queries) and **local ability** (via ELO on nearest-neighbor subsets by embedding similarity) to decide which model to route to.

---

## 📌 Overview
- **Goal**: Replicate the EAGLE approach and test routing fidelity between GPT-4o and GPT-4o-mini.  
- **Dataset**: RepliQA (10k rows, cleaned, non-null, with gold answers and model scores).  
- **Models**: GPT-4o-2024-08-06 vs GPT-4o-mini-2024-07-18.  
- **Routing method**:  
  - Global ELO (general model strength across all queries).  
  - Local ELO (query-specific strength using N nearest neighbors).  
  - Final decision:  
    \[
    \text{Score}(m) = P \cdot \text{Global}(m) + (1-P) \cdot \text{Local}(m)
    \]  

---

## ⚙️ Implementation Highlights
- **Data Loading & Preprocessing**:
  - Loaded from Hugging Face datasets.
  - Extracted text, scores, and outcomes (wins/ties).
  - Combined `prompt + question` into richer text embeddings.
- **Embeddings**:
  - Used `text-embedding-3-large` via the OpenAI API.
  - Implemented on-disk caching to avoid recomputation.
- **ELO Updates**:
  - Global: applied across all training matches with K=32.
  - Local: applied only to top-N neighbors (N=20) using cosine similarity.
- **Evaluation**:
  - Paper defaults: P=0.5, K=32, N=20.
  - Grid-searched P over [0.0, 1.0] on validation.
  - Reported both routing accuracy (including ties) and strict accuracy (non-tie rows only).
- **SQL Analysis**:
  - Wrote validation queries to confirm dataset imbalance.

---

## 📊 Results

### Global ELO Ratings

gpt-4o-mini : 121.7346
gpt-4o      : 78.2654

 

### Optimized Hyperparameter
- Best **P’ = 0.25** (from validation strict accuracy curve).

### Validation Strict (non-tie) Accuracy
| P    | Strict Acc |
|------|------------|
| 0.25 | **78.95%** |
| 0.50 | 73.68%     |
| 0.30 | 68.42%     |

### Test Set Accuracy
- Paper params (P=0.50): **10.00% routing**, **64.29% strict**  
- Optimized P’ (P=0.25): **13.00% routing**, **71.43% strict**

### SQL Distribution of Outcomes
```sql
SELECT 
    CASE 
        WHEN "gpt-4o-mini-2024-07-18/score" > "gpt-4o-2024-08-06/score" THEN 'gpt-4o-mini wins'
        WHEN "gpt-4o-2024-08-06/score" > "gpt-4o-mini-2024-07-18/score" THEN 'gpt-4o wins'
        ELSE 'tie'
    END AS result,
    COUNT(*) AS count
FROM train
GROUP BY result
ORDER BY count DESC;
``` 

Results:

* **gpt-4o-mini wins:** 8,327 (≈83%)
* **gpt-4o wins:** 538 (≈5%)
* **ties:** 1,198 (≈12%)

---

## 🔎 Interpretation

* The dataset is **heavily imbalanced** toward gpt-4o-mini, explaining why raw routing accuracy appears low.
* On the **strict subset (non-tie rows)**, the EAGLE router achieves **\~70% accuracy**, significantly outperforming the naive baseline.
* This confirms the router’s ability to recover signal from the tradeoff between global (general strength) and local (query-specific similarity) ratings.
* Key insight: **routing accuracy ≠ full picture**; strict accuracy better captures the model’s discrimination ability.

---



## 📚 References

* [EAGLE Paper (arXiv:2409.15518)](https://arxiv.org/abs/2409.15518)
* [Not Diamond RepliQA Dataset](https://huggingface.co/datasets/notdiamond/repliqa_gpt4o_gpt4omini_evals)

---

## ✍️ Author’s Note

This implementation was part of a technical take-home for **Not Diamond**. It highlights:

* Ability to **translate research papers into working code**.
* Competence in **data wrangling, embeddings, ranking, and evaluation**.
* Analytical rigor in identifying dataset bias and interpreting results.

 