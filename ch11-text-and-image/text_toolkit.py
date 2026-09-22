"""
text_toolkit.py
---------------
Global South Urban Methods — Chapter 10 (text half).
Classifying and topic-modelling urban text when you do not have a large,
clean, well-resourced corpus.

    python text_toolkit.py

Requires: scikit-learn, pandas, numpy
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================ 1. VOCABULARY
def vocab_profile(texts):
    """Basic corpus diagnostics. Run this before anything else.

    `tokens_per_type` is the number that matters in low-resource settings:
    inconsistent spelling inflates the type count, leaving fewer examples
    of each token for a model to learn from.
    """
    toks = " ".join(texts).split()
    types = set(toks)
    hapax = sum(1 for t in types if toks.count(t) == 1) if len(types) < 4000 else None
    return {"n_docs": len(texts),
            "n_tokens": len(toks),
            "n_types": len(types),
            "tokens_per_type": round(len(toks) / len(types), 2),
            "mean_doc_len": round(len(toks) / len(texts), 1),
            "hapax_share": round(hapax / len(types), 3) if hapax is not None else None}


# ========================================================== 2. CLASSIFICATION
def build_classifier(char_ngrams=False):
    """Word n-grams by default; character n-grams are far more robust to
    inconsistent orthography, which is why they matter for low-resource text."""
    if char_ngrams:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              min_df=2, sublinear_tf=True)
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return make_pipeline(vec, LogisticRegression(max_iter=2000, C=4.0))


def evaluate(texts, labels, char_ngrams=False, k=5, seed=0):
    clf = build_classifier(char_ngrams)
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    s = cross_val_score(clf, texts, labels, cv=cv, scoring="f1_macro")
    return {"f1_macro_mean": round(float(s.mean()), 3),
            "f1_macro_std": round(float(s.std()), 3)}


def learning_curve(texts, labels, sizes=(50, 100, 200, 400, 800, 1600),
                   char_ngrams=False, seed=0, repeats=3):
    """How much does performance fall as the labelled corpus shrinks?

    This is the honest way to answer 'do I have enough annotated data?' —
    and the honest way to show what a low-resource language costs you.
    """
    texts, labels = np.asarray(texts), np.asarray(labels)
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        if n > len(texts) * 0.8:
            continue
        scores = []
        for r in range(repeats):
            idx = rng.choice(len(texts), n, replace=False)
            try:
                scores.append(evaluate(texts[idx], labels[idx],
                                       char_ngrams, k=3, seed=r)["f1_macro_mean"])
            except ValueError:
                continue
        if scores:
            rows.append({"n_train": n, "f1_macro": round(float(np.mean(scores)), 3),
                         "sd": round(float(np.std(scores)), 3)})
    return pd.DataFrame(rows)


# ========================================================= 3. TOPIC MODELLING
def topic_model(texts, n_topics=5, n_top=8, seed=0):
    """LDA topics. Use when you have no labels at all and need to find out
    what a corpus is about before deciding what to annotate."""
    vec = CountVectorizer(max_df=0.55, min_df=4, stop_words="english")
    dtm = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=seed,
                                    learning_method="batch", max_iter=30)
    W = lda.fit_transform(dtm)
    names = np.array(vec.get_feature_names_out())
    topics = [{"topic": i,
               "top_terms": ", ".join(names[c.argsort()[::-1][:n_top]]),
               "share": round(float((W.argmax(1) == i).mean()), 3)}
              for i, c in enumerate(lda.components_)]
    return pd.DataFrame(topics), W, lda


def topic_label_crosstab(W, labels):
    """Do the unsupervised topics line up with known categories?
    If they do, topic modelling can substitute for hand annotation."""
    return pd.crosstab(pd.Series(W.argmax(1), name="topic"),
                       pd.Series(labels, name="label"))


if __name__ == "__main__":
    en = pd.read_csv("data/complaints_en.csv")
    lr = pd.read_csv("data/complaints_lowres.csv")
    print("English  :", vocab_profile(en.text))
    print("Low-res  :", vocab_profile(lr.text))
    print("\nEnglish  word n-grams:", evaluate(en.text, en.label))
    print("Low-res  word n-grams:", evaluate(lr.text, lr.label))
    print("Low-res  char n-grams:", evaluate(lr.text, lr.label, char_ngrams=True))
