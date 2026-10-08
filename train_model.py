"""Train sentiment + topic classifiers and export them as JSON-in-JS so the website can run them in the browser."""
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("zaiqa_reviews.csv")
tr, te = train_test_split(df, test_size=.2, random_state=42, stratify=df.topic)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=4000).fit(tr.review_text)
Xtr, Xte = vec.transform(tr.review_text), vec.transform(te.review_text)

model = {"vocab": {k: int(v) for k, v in vec.vocabulary_.items()}, "idf": [round(float(x), 4) for x in vec.idf_], "acc": {}}
for t in ("sentiment", "topic"):
    clf = LogisticRegression(C=5, max_iter=2000).fit(Xtr, tr[t])
    model[t] = {"classes": clf.classes_.tolist(), "coef": [[round(float(x), 3) for x in row] for row in clf.coef_],
                "intercept": [round(float(x), 4) for x in clf.intercept_]}
    model["acc"][t] = round(float(clf.score(Xte, te[t])), 3)
    if t == "sentiment": pd.DataFrame(clf.predict_proba(Xte[:200]), columns=clf.classes_).assign(text=te.review_text[:200].values).to_csv("/tmp/parity.csv", index=False)

open("model.js", "w").write("const MODEL = " + json.dumps(model, separators=(",", ":")) + ";")
reviews = df[["date", "branch", "platform", "rating", "review_text"]].to_dict("records")
open("reviews.js", "w", encoding="utf-8").write("const REVIEWS = " + json.dumps(reviews, ensure_ascii=False, separators=(",", ":")) + ";")
print("accuracy:", model["acc"])
