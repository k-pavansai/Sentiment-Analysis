import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sentiment_data.csv")
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
df = df[["Comment", "Sentiment"]].dropna().copy()
df["Comment"] = df["Comment"].astype(str).str.strip()
df = df[df["Comment"] != ""]
df["Sentiment"] = pd.to_numeric(df["Sentiment"], errors="coerce")
df = df.dropna(subset=["Sentiment"])
df["Sentiment"] = df["Sentiment"].astype(int)
df = df.drop_duplicates(subset=["Comment", "Sentiment"])

# Label meanings follow the supplied experiment code.
labels = [0, 1, 2]
label_names = ["Negative", "Neutral", "Positive"]
if set(df["Sentiment"].unique()) != set(labels):
    raise ValueError("Expected Sentiment labels 0, 1 and 2.")

X_train, X_test, y_train, y_test = train_test_split(
    df["Comment"], df["Sentiment"], test_size=0.20,
    random_state=42, stratify=df["Sentiment"]
)
vectorizer = TfidfVectorizer(
    max_features=5000, min_df=3, max_df=0.95, sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Linear SVM": LinearSVC(max_iter=1000, random_state=42)
}
results = []
best_name, best_f1 = None, -1
best_pred = None

for name, model in models.items():
    start = time.time()
    model.fit(X_train_tfidf, y_train)
    pred = model.predict(X_test_tfidf)
    elapsed = time.time() - start
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "Macro F1": macro_f1,
        "Training Time (seconds)": elapsed
    })
    print("\n" + name)
    print(classification_report(
        y_test, pred, labels=labels, target_names=label_names, zero_division=0
    ))
    if macro_f1 > best_f1:
        best_name, best_f1, best_pred = name, macro_f1, pred

comparison = pd.DataFrame(results).sort_values("Macro F1", ascending=False)
print("\nMODEL COMPARISON")
print(comparison.round(4).to_string(index=False))
print("\nBest model based on Macro F1:", best_name)

fig, ax = plt.subplots(figsize=(8, 5))
comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "Macro F1"]].plot(
    kind="bar", ax=ax, ylim=(0, 1)
)
ax.set_title("Sentiment Analysis - Model Comparison")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "model_comparison.png"), dpi=200)
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, labels=labels, display_labels=label_names, cmap="Blues"
)
plt.title("Best Sentiment Model - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=200)
plt.show()
