import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from random import randrange
from stemming.porter2 import stem
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def load_data_manually(path):
    sentences = []
    labels = []
    label_to_index = {'Metal': 0,
                      'Pop': 1,
                      'Jazz': 2,
                      'Rock': 3,
                      'Folk': 4}
    with open(path, encoding='utf-8', mode='r') as in_file:
        for line in in_file:
            vals = line.strip().split(',')
            sentences.append(vals[4])
            labels.append(label_to_index[vals[5]])

    sentences = [[stem(word) for word in sentence.split(" ")] for sentence in sentences]
    sentences = [" ".join(sentence) for sentence in sentences]
    return sentences, labels


def evaluate(y_true, y_pred):
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], digits=4))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    print(f"accuracy: {acc}, f1: {f1}, precision: {precision}, recall: {recall}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    print(cm)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['Metal', 'Pop', 'Jazz', 'Rock', 'Folk'])
    ax.yaxis.set_ticklabels(['Metal', 'Pop', 'Jazz', 'Rock', 'Folk'])
    plt.show()


def randomClassifier(length):
    y_pred = []
    for i in range(length):
        y_pred.append(randrange(5))
    return y_pred


if __name__ == "__main__":
    train_s, train_l = load_data_manually('train_english.csv')
    # valid_s, valid_l = load_data_manually('valid/valid.csv')
    test_s, test_l = load_data_manually('test_english.csv')

    countVec = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=10000)

    # Creating tf-idf vector for the documents

    x_trainCV = countVec.fit_transform(train_s)

    x_testCV = countVec.transform(test_s)

    # converting the vectors into array to use further
    x_train = x_trainCV.toarray()
    x_test = x_testCV.toarray()

    print("x_train: %s, x_test: %s, y_train: %s, y_test: %s" % (len(x_train), len(x_test), len(train_l), len(test_l)))


    # svm
    print("SVM classifier")
    svm = Pipeline([('vect', TfidfVectorizer(analyzer=lambda x: x)), ('clf', SVC(kernel='rbf'),)])
    svm = svm.fit(x_train, train_l)
    accuracy = svm.score(x_test, test_l)
    print("SVM: accuracy for tf-idf %s" % (accuracy))

    print("Multinomial Naive Bayes Classifier")
    mnb = MultinomialNB()
    mnb.fit(x_train, train_l)
    accuracy = mnb.score(x_test, test_l)
    print("accuracy for multinomial naive bayes: %s" % (accuracy))


    print("Logistic Regression Classifier")
    lr = LogisticRegression(solver="liblinear", multi_class="ovr")
    lr.fit(x_train, train_l)
    print("accuracy for LogisticRegression: %s" % (lr.score(x_test, test_l)))


    # randomClassifier
    print('Random Classifier')
    y_random = randomClassifier(len(test_l))
    evaluate(test_l, y_random)

    print("Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=10, max_features="sqrt").fit(x_train, train_l)
    accuracy = rf.score(x_test, test_l)
    print("accuracy for Random Forest: %s" % (accuracy))

    print("Bernoulli Naive Bayes Classifier")
    bnb = BernoulliNB()
    bnb.fit(x_train, train_l)
    accuracy = bnb.score(x_test, test_l)
    print("accuracy for bernoulli naive bayes: %s" % (accuracy))

    print("Decision Tree Classifier")
    dt = DecisionTreeClassifier()
    dt.fit(x_train, train_l)
    print("accuracy for Decision Tree: %s" % (dt.score(x_test, test_l)))

    print("Multi Layer Perceptron Classifier")
    # Training and Testing on SCikit Neural Network library
    neural = MLPClassifier()
    neural.fit(x_train, train_l)
    accuracy = neural.score(x_test, test_l)
    print("accuracy for Neural Network: %s" % (accuracy))
