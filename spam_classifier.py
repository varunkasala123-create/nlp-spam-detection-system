import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ----------------- CREATE DATA ----------------- #

spam_words = ["win", "free", "cash", "urgent", "offer", "credit", "prize"]
ham_words = ["meeting", "project", "schedule", "team", "report", "deadline", "update"]

np.random.seed(0)

def make_email(spam=True):
    words = []
    base = spam_words if spam else ham_words

    for _ in range(np.random.randint(5, 15)):
        if np.random.rand() < 0.7:
            words.append(np.random.choice(base))
        else:
            words.append(np.random.choice(spam_words + ham_words))

    return " ".join(words)

emails = []
labels = []

for _ in range(500):
    emails.append(make_email(spam=True))
    labels.append(1)

for _ in range(500):
    emails.append(make_email(spam=False))
    labels.append(0)

df = pd.DataFrame({
    "text": emails,
    "label": labels
})

# ----------------- SPLIT ----------------- #

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# ----------------- VECTORIZE TEXT ----------------- #

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------- TRAIN MODEL ----------------- #

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ----------------- EVALUATE ----------------- #

preds = model.predict(X_test_vec)
probs = model.predict_proba(X_test_vec)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, preds))

auc = roc_auc_score(y_test, probs)
print("ROC AUC:", auc)

# ----------------- ROC CURVE ----------------- #

fpr, tpr, _ = roc_curve(y_test, probs)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# -------- MOST IMPORTANT WORDS -------- #

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_spam = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:10]
top_ham = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:10]

print("\nTop spam-indicating words:")
for w,c in top_spam:
    print(w, round(c,2))

print("\nTop ham-indicating words:")
for w,c in top_ham:
    print(w, round(c,2))

