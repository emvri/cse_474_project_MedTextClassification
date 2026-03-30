import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import preprocess
from features import get_bow, get_tfidf
from models import train_nb
from models import train_lr



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
# sklearn models require numeric targets, here we convert text labels
# like BACKGROUND / METHODS / RESULTS into integer values.
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
nb_bow = train_nb(X_train_bow, y_train)
y_pred_bow = nb_bow.predict(X_test_bow)

print(" Naive Bayes + Bag-of-Words ")
# Evaluation metric: accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_bow))

# Same evaluation style as TDD_march05:
# classification_report gives precision, recall, and F1-score
print(classification_report(y_test, y_pred_bow, target_names=le.classes_))

# Same confusion-matrix idea from lecture:
# helps us see which labels are being confused with each other
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bow))

# Save results
with open("results/metrics.txt", "w") as f:
    f.write(" Naive Bayes + Bag-of-Words \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_bow):.4f}\n\n")
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
nb_tfidf = train_nb(X_train_tfidf, y_train)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

print("\n Naive Bayes + TF-IDF ")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tfidf))

with open("results/metrics.txt", "a") as f:
    f.write("\n\n Naive Bayes + TF-IDF \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf):.4f}\n\n")
    f.write(classification_report(y_test, y_pred_tfidf, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_tfidf)))



# Experiment 3: Bag-of-Words + Logistic Regression
# -

# Same Bag-of-Words feature representation, now using Logistic Regression
# so we can compare model performance fairly on the same features.
lr_bow = train_lr(X_train_bow, y_train)
y_pred_lr_bow = lr_bow.predict(X_test_bow)

print("\n Logistic Regression + Bag-of-Words ")
print("Accuracy:", accuracy_score(y_test, y_pred_lr_bow))
print(classification_report(y_test, y_pred_lr_bow, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_bow))

with open("results/metrics.txt", "a") as f:
    f.write("\n\n Logistic Regression + Bag-of-Words \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr_bow):.4f}\n\n")
    f.write(classification_report(y_test, y_pred_lr_bow, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_lr_bow)))


# Experiment 4: TF-IDF + Logistic Regression
# -

# Same Logistic Regression model, but now using TF-IDF features
# to compare how feature weighting affects performance.
lr_tfidf = train_lr(X_train_tfidf, y_train)
y_pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)

print("\n Logistic Regression + TF-IDF ")
print("Accuracy:", accuracy_score(y_test, y_pred_lr_tfidf))
print(classification_report(y_test, y_pred_lr_tfidf, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_tfidf))

with open("results/metrics.txt", "a") as f:
    f.write("\n\n Logistic Regression + TF-IDF \n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr_tfidf):.4f}\n\n")
    f.write(classification_report(y_test, y_pred_lr_tfidf, target_names=le.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_lr_tfidf)))