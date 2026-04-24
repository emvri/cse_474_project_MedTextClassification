import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import preprocess
from features import get_bow, get_tfidf
from models import train_nb
from models import train_lr
from models import train_svm
import time
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip blank lines
            if not line:
                continue

            # skip abstract id lines like ###24562799
            if line.startswith("###"):
                continue

            # only process real label + text lines
            parts = line.split("\t")
            if len(parts) == 2:
                label, text = parts
                data.append((label, text))

    return pd.DataFrame(data, columns=["label", "text"])


# Load training and test sets
# - 
# For this project, we use the provided PubMed train/test split
# instead of train_test_split from lecture.
# This is more appropriate because the dataset already comes with
# predefined splits for model evaluation.
train_df = load_data("data/train.txt")
test_df = load_data("data/test.txt")

# Preprocess text
# -
# Clean raw input before modeling.

train_df["clean"] = train_df["text"].apply(preprocess)
test_df["clean"] = test_df["text"].apply(preprocess)

# Encode labels
# -
# sklearn models require numeric targets
# Convert labels into integer values
le = LabelEncoder()
y_train = le.fit_transform(train_df["label"])
y_test = le.transform(test_df["label"])


# Experiment 1: Bag-of-Words + Naive Bayes
# -
# Convert data into numerical features before training.
X_train_bow, bow_vec = get_bow(train_df["clean"])

# Fit the vectorizer on training data only, then transform test data using the same learned vocabulary.
X_test_bow = bow_vec.transform(test_df["clean"])

# Same training pattern used in class:
# define model -> fit on training data -> predict on test data
start = time.time()
nb_bow = train_nb(X_train_bow, y_train)
train_time = time.time() - start

start = time.time()
y_pred_bow = nb_bow.predict(X_test_bow)
inference_time = time.time() - start

print(" Naive Bayes + Bag-of-Words ")
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print("Train Time:", round(train_time, 4))
print("Inference Time:", round(inference_time, 4))




# Same evaluation style as TDD_march05:
# classification_report gives precision, recall, and F1-score
print(classification_report(y_test, y_pred_bow, target_names=le.classes_))

# Same confusion-matrix idea from lecture:
# helps us see which labels are being confused with each other
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bow))

# Save results
with open("results/metrics.txt", "w") as f:
    f.write(" Naive Bayes + Bag-of-Words \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_bow):.4f}\n")
    f.write(f"Train Time: {train_time:.4f}\n")
    f.write(f"Inference Time: {inference_time:.4f}\n\n")
    f.write(classification_report(y_test, y_pred_bow, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_bow)))


# Experiment 2: TF-IDF + Naive Bayes
# - 
# Alternative feature representation:
# TF-IDF still uses word frequency, but downweights very common words.
# This connects to lecture discussion of feature representation and
# how the choice of features can affect model performance.
X_train_tfidf, tfidf_vec = get_tfidf(train_df["clean"])
X_test_tfidf = tfidf_vec.transform(test_df["clean"])

# Same training/evaluation pipeline as above so comparison stays fair
start = time.time()
nb_tfidf = train_nb(X_train_tfidf, y_train)
train_time = time.time() - start

start = time.time()
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
inference_time = time.time() - start

print("\n Naive Bayes + TF-IDF ")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print("Train Time:", round(train_time, 4))
print("Inference Time:", round(inference_time, 4))
print(classification_report(y_test, y_pred_tfidf, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tfidf))

with open("results/metrics.txt", "a") as f:
    f.write("\n\n Naive Bayes + TF-IDF \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf):.4f}\n")
    f.write(f"Train Time: {train_time:.4f}\n")
    f.write(f"Inference Time: {inference_time:.4f}\n\n")    
    f.write(classification_report(y_test, y_pred_tfidf, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_tfidf)))



# Experiment 3: Bag-of-Words + Logistic Regression
# -
# Same Bag-of-Words feature representation, now using Logistic Regression
# so we can compare model performance fairly on the same features.
start = time.time()
lr_bow = train_lr(X_train_bow, y_train)
train_time = time.time() - start

start = time.time()
y_pred_lr_bow = lr_bow.predict(X_test_bow)
inference_time = time.time() - start

print("\n Logistic Regression + Bag-of-Words ")
print("Accuracy:", accuracy_score(y_test, y_pred_lr_bow))
print("Train Time:", round(train_time, 4))
print("Inference Time:", round(inference_time, 4))
print(classification_report(y_test, y_pred_lr_bow, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_bow))

with open("results/metrics.txt", "a") as f:
    f.write("\n\n Logistic Regression + Bag-of-Words \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr_bow):.4f}\n")
    f.write(f"Train Time: {train_time:.4f}\n")
    f.write(f"Inference Time: {inference_time:.4f}\n\n")    
    f.write(classification_report(y_test, y_pred_lr_bow, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_lr_bow)))


# Experiment 4: TF-IDF + Logistic Regression
# -
# Same Logistic Regression model, but now using TF-IDF features
# to compare how feature weighting affects performance.
start = time.time()
lr_tfidf = train_lr(X_train_tfidf, y_train)
train_time = time.time() - start

start = time.time()
y_pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)
inference_time = time.time() - start

print("\n Logistic Regression + TF-IDF ")
print("Accuracy:", accuracy_score(y_test, y_pred_lr_tfidf))
print("Train Time:", round(train_time, 4))
print("Inference Time:", round(inference_time, 4))
print(classification_report(y_test, y_pred_lr_tfidf, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_tfidf))

with open("results/metrics.txt", "a") as f:
    f.write("\n\n Logistic Regression + TF-IDF \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr_tfidf):.4f}\n")
    f.write(f"Train Time: {train_time:.4f}\n")
    f.write(f"Inference Time: {inference_time:.4f}\n\n")    
    f.write(classification_report(y_test, y_pred_lr_tfidf, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_lr_tfidf)))



# Experiment 5: Bag-of-Words + SVM
# -
# Same Bag-of-Words feature representation, now using SVM
# so we can compare model performance fairly on the same features.
print("Training SVM...")
start = time.time()
svm_bow = train_svm(X_train_bow, y_train)
train_time = time.time() - start
print("Done!")

start = time.time()
y_pred_svm_bow = svm_bow.predict(X_test_bow)
inference_time = time.time() - start

print("\n SVM + Bag-of-Words ")
print("Accuracy:", accuracy_score(y_test, y_pred_svm_bow))
print("Train Time:", round(train_time, 4))
print("Inference Time:", round(inference_time, 4))
print(classification_report(y_test, y_pred_svm_bow, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm_bow))

with open("results/metrics.txt", "a") as f:
    f.write("\n\n SVM + Bag-of-Words \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_svm_bow):.4f}\n")
    f.write(f"Train Time: {train_time:.4f}\n")
    f.write(f"Inference Time: {inference_time:.4f}\n\n")
    f.write(classification_report(y_test, y_pred_svm_bow, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_svm_bow)))


# Experiment 6: TF-IDF + SVM
# -
# Same SVM model, but now using TF-IDF features
# to compare how feature weighting affects performance.

start = time.time()
svm_tfidf = train_svm(X_train_tfidf, y_train)
train_time = time.time() - start

start = time.time()
y_pred_svm_tfidf = svm_tfidf.predict(X_test_tfidf)
inference_time = time.time() - start


print("\n SVM + TF-IDF ")
print("Accuracy:", accuracy_score(y_test, y_pred_svm_tfidf))
print("Train Time:", round(train_time, 4))
print("Inference Time:", round(inference_time, 4))
print(classification_report(y_test, y_pred_svm_tfidf, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm_tfidf))

with open("results/metrics.txt", "a") as f:
    f.write("\n\n SVM + TF-IDF \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_svm_tfidf):.4f}\n")
    f.write(f"Train Time: {train_time:.4f}\n")
    f.write(f"Inference Time: {inference_time:.4f}\n\n")
    f.write(classification_report(y_test, y_pred_svm_tfidf, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_svm_tfidf)))


models = [
    "NB (BoW)", "NB (TF-IDF)",
    "LR (BoW)", "LR (TF-IDF)",
    "SVM (BoW)", "SVM (TF-IDF)"
]

accuracies = [
    0.7726, 0.7650,
    0.8124, 0.8135,
    0.8059, 0.8063
]

train_times = [
    0.3383, 0.3034,
    240.2361, 130.4513,
    393.1530, 173.8920
]

plt.figure()

for i in range(len(models)):
    plt.scatter(train_times[i], accuracies[i])
    plt.text(train_times[i], accuracies[i], models[i])

plt.xlabel("Training Time (seconds)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Training Time Tradeoff")

plt.savefig("results/accuracy_vs_time.png")
plt.show()


# Best model confusion matrix (Logistic Regression + TF-IDF)
cm = confusion_matrix(y_test, y_pred_lr_tfidf)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap (Logistic Regression + TF-IDF)")

plt.savefig("results/confusion_matrix_heatmap.png")
plt.show()
