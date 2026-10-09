"""Topic discovery.

Stage A  (discovery)  BERTopic: MiniLM sentence embeddings -> UMAP(5d) -> HDBSCAN -> class-based TF-IDF.
Stage B  (taxonomy)   Micro-topics are mapped to a voter-issue taxonomy (results/topic_issue_map.csv). The
                      map is produced by an LLM annotator (Claude) reading each topic's c-TF-IDF terms and
                      representative documents, and is cross-validated against a fully automatic
                      embedding-anchor assignment (Cohen's kappa reported).
Baselines             LDA (gensim) and NMF (sklearn) on the same corpus; NPMI / C_v coherence, topic diversity
                      and seed-stability, following the evaluation agenda of Kherwa & Bansal (2019).
"""
import json, re, numpy as np, pandas as pd
from .config import PROC, RES, SEED

EMB = PROC / "embeddings.npz"
STOP_EXTRA = {"user", "midterms", "midterm", "election", "elections", "congress", "democrats", "democrat",
              "republicans", "republican", "voters", "voter", "amp", "just", "like", "people", "don", "know",
              "think", "going", "want", "say", "said", "says", "get", "got", "need", "make", "really", "right",
              "way", "time", "year", "years", "today", "new", "news", "did", "does", "doing", "let", "ll", "ve",
              "re", "im", "it", "thats", "dont", "cant", "didnt", "isnt", "won", "lot", "thing", "things"}


def stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return sorted(set(ENGLISH_STOP_WORDS) | STOP_EXTRA)


def embed(c):
    """Sentence embeddings cached by doc_id (incremental: only new documents are encoded)."""
    ids = c.doc_id.values
    cache = {}
    if EMB.exists():
        z = np.load(EMB, allow_pickle=True); cache = dict(zip(z["ids"], z["emb"]))
    todo = [i for i, d in enumerate(ids) if d not in cache]
    if todo:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
        new = m.encode(c.text.iloc[todo].tolist(), batch_size=512, show_progress_bar=True, normalize_embeddings=True)
        for i, v in zip(todo, new):
            cache[ids[i]] = v.astype(np.float32)
        np.savez(EMB, ids=np.array(list(cache), dtype=object), emb=np.stack(list(cache.values())))
    return np.stack([cache[d] for d in ids])


def fit_bertopic(c, e):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    mcs = max(150, len(c) // 1500)
    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED, low_memory=True),
        hdbscan_model=HDBSCAN(min_cluster_size=mcs, min_samples=10, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=False),
        vectorizer_model=CountVectorizer(stop_words=stopwords(), ngram_range=(1, 2), min_df=10, max_df=0.5),
        calculate_probabilities=False, verbose=True)
    topics, _ = model.fit_transform(c.text.tolist(), e)
    return model, np.array(topics)


def topic_table(model, c, topics, e, k_docs=4):
    rows = []
    for t in sorted(set(topics)):
        if t == -1:
            continue
        idx = np.where(topics == t)[0]
        cen = e[idx].mean(0); cen /= np.linalg.norm(cen)
        top = idx[np.argsort(-(e[idx] @ cen))[:k_docs]]
        words = [w for w, _ in model.get_topic(t)][:10]
        sub = c.iloc[idx]
        rows.append({"topic": t, "n": len(idx), "share_news": (sub.channel == "news").mean(),
                     "terms": ", ".join(words), "examples": " || ".join(c.text.iloc[top].str[:160])})
    return pd.DataFrame(rows)


# ---------------- issue taxonomy ----------------
ISSUE_ANCHORS = {  # natural-language definitions used ONLY by the automatic cross-check
    "Economy & cost of living": "inflation prices groceries cost of living affordability wages jobs recession gas prices economy stock market",
    "Tariffs & trade": "tariffs trade war import taxes china tariffs supreme court tariff ruling",
    "Health care": "health care medicaid cuts affordable care act obamacare premiums subsidies insurance hospitals",
    "Immigration & ICE": "immigration ICE raids deportation border detention migrants",
    "Democracy & rule of law": "democracy authoritarian rule of law election integrity voting rights stealing election courts constitution",
    "Foreign policy & Iran war": "iran war military strikes foreign policy israel ukraine russia troops",
    "Epstein files & corruption": "epstein files corruption scandal cover up investigation crypto bribery",
    "Redistricting & gerrymandering": "redistricting gerrymandering maps texas california proposition seats",
    "Government shutdown & budget": "government shutdown budget spending bill continuing resolution debt deficit",
    "Abortion & reproductive rights": "abortion roe reproductive rights planned parenthood",
    "Crime & public safety": "crime public safety national guard police cities",
    "Social Security & safety net": "social security snap food stamps benefits seniors safety net",
}


def auto_map(model, topics, e, thresh=0.25):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
    names = list(ISSUE_ANCHORS)
    A = m.encode([ISSUE_ANCHORS[k] for k in names], normalize_embeddings=True)
    out = {}
    for t in sorted(set(topics) - {-1}):
        idx = np.where(topics == t)[0]
        cen = e[idx].mean(0); cen /= np.linalg.norm(cen)
        s = A @ cen
        out[t] = names[int(s.argmax())] if s.max() >= thresh else "Other / non-issue"
    return out


# ---------------- baselines + evaluation ----------------
TOK = re.compile(r"[a-z][a-z']{2,}")


def tokenize(texts, stops):
    S = set(stops)
    return [[w for w in TOK.findall(t.lower()) if w not in S] for t in texts]


def coherence(topics_words, toks, dictionary):
    from gensim.models.coherencemodel import CoherenceModel
    tw = [[w for w in ws if w in dictionary.token2id][:10] for ws in topics_words]
    tw = [ws for ws in tw if len(ws) >= 3]
    out = {}
    for m in ("c_npmi", "c_v"):
        out[m] = CoherenceModel(topics=tw, texts=toks, dictionary=dictionary, coherence=m, processes=1).get_coherence()
    allw = [w for ws in tw for w in ws]
    out["diversity"] = len(set(allw)) / max(1, len(allw))
    return out


def jaccard_stability(runs):
    """Mean best-match Jaccard between top-10 word sets of the first run and every other run."""
    base = [set(t) for t in runs[0]]; sims = []
    for r in runs[1:]:
        other = [set(t) for t in r]
        sims.append(np.mean([max(len(a & b) / len(a | b) for b in other) for a in base]))
    return float(np.mean(sims)), float(np.std(sims))


def baselines(c, bert_words, n=40000, ks=(10, 20, 30), seeds=(0, 1, 2, 3, 4)):
    from gensim.corpora import Dictionary
    from gensim.models import LdaMulticore
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    s = c.sample(min(n, len(c)), random_state=SEED)
    stops = stopwords(); toks = tokenize(s.text, stops)
    d = Dictionary(toks); d.filter_extremes(no_below=10, no_above=0.4)
    bow = [d.doc2bow(t) for t in toks]
    tf = TfidfVectorizer(stop_words=stops, min_df=10, max_df=0.4, token_pattern=r"[a-z][a-z']{2,}")
    X = tf.fit_transform(s.text.str.lower()); vocab = np.array(tf.get_feature_names_out())
    rows, stab = [], {}
    for k in ks:
        lda_runs, nmf_runs = [], []
        for sd in seeds:
            lda = LdaMulticore(bow, id2word=d, num_topics=k, passes=4, random_state=sd, workers=4)
            lda_runs.append([[w for w, _ in lda.show_topic(i, 10)] for i in range(k)])
            nmf = NMF(k, init="nndsvda" if sd == 0 else "random", random_state=sd, max_iter=400).fit(X)
            nmf_runs.append([list(vocab[np.argsort(-r)[:10]]) for r in nmf.components_])
        for name, runs in (("LDA", lda_runs), ("NMF", nmf_runs)):
            co = coherence(runs[0], toks, d)
            st = jaccard_stability(runs)
            rows.append({"model": name, "K": k, **co, "stability_jaccard": st[0], "stability_sd": st[1]})
    co = coherence(bert_words, toks, d)
    rows.append({"model": "BERTopic", "K": len(bert_words), **co, "stability_jaccard": np.nan, "stability_sd": np.nan})
    return pd.DataFrame(rows)


def main(run_baselines=True):
    c = pd.read_parquet(PROC / "corpus.parquet")
    e = embed(c)
    model, topics = fit_bertopic(c, e)
    c["topic"] = topics
    c[["doc_id", "topic"]].to_parquet(PROC / "doc_topics.parquet")
    tt = topic_table(model, c, topics, e)
    am = auto_map(model, topics, e)
    tt["auto_issue"] = tt.topic.map(am)
    tt.to_csv(RES / "topic_info.csv", index=False)
    np.save(PROC / "topic_centroids.npy", np.stack([
        (lambda v: v / np.linalg.norm(v))(e[topics == t].mean(0)) for t in tt.topic]))
    print(f"{len(tt)} topics; outlier share {np.mean(topics == -1):.2%}")
    if run_baselines:
        bw = [[w for w, _ in model.get_topic(t)][:10] for t in tt.sort_values("n", ascending=False).topic.head(30)]
        run_baselines(); return
        b.to_csv(RES / "topic_model_comparison.csv", index=False); print(b)


if __name__ == "__main__":
    main()


def run_baselines():
    """Baselines can be (re)run without refitting BERTopic: BERTopic topics are read from topic_info.csv."""
    c = pd.read_parquet(PROC / "corpus.parquet")
    info = pd.read_csv(RES / "topic_info.csv").sort_values("n", ascending=False).head(30)
    bw = [[w for w in t.split(", ") if " " not in w][:10] for t in info.terms]  # unigrams only, comparable to LDA/NMF
    b = baselines(c, bw)
    b.to_csv(RES / "topic_model_comparison.csv", index=False); print(b.round(3).to_string())


def bertopic_stability(n=30000, seeds=(7, 11, 23)):
    """Seed stability of the BERTopic pipeline on a fixed subsample: adjusted Rand index between partitions
    (documents clustered in both runs) and best-match Jaccard of top-10 c-TF-IDF terms."""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import adjusted_rand_score
    c = pd.read_parquet(PROC / "corpus.parquet").sample(n, random_state=SEED).reset_index(drop=True)
    e = embed(c); parts, words = [], []
    for sd in seeds:
        m = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=sd),
                     hdbscan_model=HDBSCAN(min_cluster_size=60, min_samples=10),
                     vectorizer_model=CountVectorizer(stop_words=stopwords(), min_df=5, max_df=0.5))
        t, _ = m.fit_transform(c.text.tolist(), e); t = np.array(t); parts.append(t)
        words.append([[w for w, _ in m.get_topic(k)][:10] for k in sorted(set(t) - {-1})])
    ari = [adjusted_rand_score(parts[0][(parts[0] != -1) & (p != -1)], p[(parts[0] != -1) & (p != -1)]) for p in parts[1:]]
    jac = jaccard_stability(words)
    out = {"ari_mean": float(np.mean(ari)), "jaccard_mean": jac[0], "jaccard_sd": jac[1],
           "n_topics": [len(w) for w in words]}
    json.dump(out, open(RES / "bertopic_stability.json", "w"), indent=2); print(out)
