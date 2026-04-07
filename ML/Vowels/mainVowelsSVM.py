import json
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from wav2vec import cutvowel, wav2vec

WAV_FILE  = "beppo.wav"
JSON_FILE = "Edu.json"

with open(JSON_FILE) as f:
    data = json.load(f)

labels  = []
vectors = []

for entry in data:
    Fs, cut = cutvowel(WAV_FILE, entry["start"], entry["end"])
    vec = wav2vec(cut, Fs)
    labels.append(entry["vocal"])
    vectors.append(vec)

X = np.array(vectors)
y = np.array(labels)

print(f"Dataset: {X.shape[0]} samples  |  Classes: {np.unique(y)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm = SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovr")

loo    = LeaveOneOut()
scores = cross_val_score(svm, X_scaled, y, cv=loo, scoring="accuracy")

print(f"\nLOO Accuracy: {scores.mean()*100:.1f}%  "
      f"(correct: {scores.sum():.0f}/{len(scores)})")

svm.fit(X_scaled, y)
y_pred = svm.predict(X_scaled)

print("\nClassification report (train set):")
print(classification_report(y, y_pred, target_names=sorted(np.unique(y))))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y, y_pred,
    display_labels=sorted(np.unique(y)),
    ax=ax,
    colorbar=False,
)
ax.set_title("Confusion Matrix (train) — SVM RBF")
plt.tight_layout()
plt.savefig("confusion_matrix_svm.png", dpi=120)
plt.show()
print("Confusion matrix saved -> confusion_matrix_svm.png")

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

f3_mean = np.full(xx.ravel().shape, X_scaled[:, 2].mean())
grid    = np.c_[xx.ravel(), yy.ravel(), f3_mean]
Z       = svm.predict(grid).reshape(xx.shape)

label_to_int = {v: i for i, v in enumerate(sorted(np.unique(y)))}
Z_int        = np.vectorize(label_to_int.get)(Z)

colors_map = {"A": "red", "E": "blue", "I": "green", "O": "orange", "U": "purple"}

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z_int, alpha=0.25,
            levels=np.arange(-0.5, len(label_to_int)), cmap="tab10")

for vocal in sorted(np.unique(y)):
    mask = y == vocal
    ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
               label=vocal, color=colors_map[vocal], edgecolors="k", s=60)

ax.set_xlabel("F1 (normalitzat)")
ax.set_ylabel("F2 (normalitzat)")
ax.set_title("Fronteres de decisió SVM — F1 vs F2")
ax.legend()
plt.tight_layout()
plt.savefig("svm_decision_boundary.png", dpi=120)
plt.show()
print("Decision boundary saved -> svm_decision_boundary.png")

def predict_vowel(wav_path: str, start: float, end: float) -> str:
    Fs, cut = cutvowel(wav_path, str(start), str(end))
    vec     = wav2vec(cut, Fs)
    vec_s   = scaler.transform([vec])
    return svm.predict(vec_s)[0]


idx   = 5
entry = data[idx]
pred  = predict_vowel(WAV_FILE, entry["start"], entry["end"])
print(f"\nSingle prediction — idx={idx}  real='{entry['vocal']}'  pred='{pred}'  "
      f"({'CORRECTE' if pred == entry['vocal'] else 'ERROR'})")
