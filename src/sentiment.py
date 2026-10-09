"""Document-level sentiment.

Primary: cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa fine-tuned on ~124M tweets + TweetEval);
         score = P(positive) - P(negative) in [-1, 1]. Chosen over FinBERT because the corpus is political
         social/news text, not financial disclosures (Du et al. 2025, sec. 3.1, on domain match).
Robustness: VADER compound (lexicon/rule-based). Agreement between the two is reported at document and
         index level; an index that only exists under one model is not trusted.
"""
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from .config import PROC

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUT = PROC / "sentiment.parquet"


def roberta(texts, bs=256):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).to("mps").eval()
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    out = np.zeros((len(texts), 3), np.float32)
    order = np.argsort([len(t) for t in texts])  # length-bucketing -> ~3x faster
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), bs), desc="roberta"):
            ix = order[i:i + bs]
            enc = tok([texts[j] for j in ix], padding=True, truncation=True, max_length=128, return_tensors="pt").to("mps")
            out[ix] = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
    return out[:, lab["negative"]], out[:, lab["neutral"]], out[:, lab["positive"]]


def vader(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    return np.array([a.polarity_scores(t)["compound"] for t in tqdm(texts, desc="vader")], np.float32)


def main():
    c = pd.read_parquet(PROC / "corpus.parquet")
    neg, neu, pos = roberta(c.text.tolist())
    s = pd.DataFrame({"doc_id": c.doc_id, "p_neg": neg, "p_neu": neu, "p_pos": pos,
                      "sent": pos - neg, "vader": vader(c.text.tolist())})
    s.to_parquet(OUT)
    print(s.describe().T)
    print("doc-level Spearman(roberta, vader) =", round(s[["sent", "vader"]].corr("spearman").iloc[0, 1], 3))


if __name__ == "__main__":
    main()
