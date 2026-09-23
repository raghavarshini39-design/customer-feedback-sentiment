import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_feedback.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_feedback"] = df["feedback"].apply(clean_text)

# 1) Sentiment distribution
counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
ax = counts.plot(kind="bar", title="Customer Feedback Sentiment Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(OUT / "sentiment_distribution.png", dpi=180)
plt.show()
plt.close()

# 2) Review text length by sentiment
df["text_length"] = df["clean_feedback"].str.split().str.len()
ax = df.boxplot(column="text_length", by="sentiment", grid=False)
plt.suptitle("")
ax.set_title("Review Length by Sentiment")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Words")
plt.tight_layout()
plt.savefig(OUT / "review_length_by_sentiment.png", dpi=180)
plt.show()
plt.close()

# 3) Train a sentiment prediction model
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_feedback"],
    df["sentiment"],
    test_size=0.25,
    random_state=42,
    stratify=df["sentiment"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, pred))

# 4) Confusion matrix
cm = confusion_matrix(y_test, pred, labels=["Positive", "Neutral", "Negative"])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Positive", "Neutral", "Negative"]
)
disp.plot()
plt.title("Sentiment Prediction Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=180)
plt.show()
plt.close()

# 5) Save predicted test results
pred_df = pd.DataFrame({
    "feedback": X_test.values,
    "actual_sentiment": y_test.values,
    "predicted_sentiment": pred
})
pred_df.to_csv(OUT / "predictions.csv", index=False)

with open(OUT / "model_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Model: TF-IDF + Logistic Regression\n")
    f.write(f"Test set size: {len(y_test)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(classification_report(y_test, pred))
