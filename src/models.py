from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_nb(X_train, y_train):
    # Naive Bayes classifier
    # Assumes word independence (covered in lecture)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model



def train_lr(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = LinearSVC(max_iter=1000)
    model.fit(X_train, y_train)
    return model