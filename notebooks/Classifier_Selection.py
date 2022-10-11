from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
# Bron: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


data = pd.read_csv(r'C:\Users\Elske\NLP local\aapl_us_equities_news_prep_text_consol_text_html_inval_upper_char_lemmas_stopwords.csv')

#vectorizer = TfidfVectorizer(max_features = 25) # Alles wordt negatief predict
vectorizer = CountVectorizer(max_features = 25) # Mooiere verdeling, slechtere accuracy

x_train, x_test, y_train, y_test = train_test_split(data['text'], data['target'], random_state= 42)

# TF-IDF matrix voor train en test data
X_train_tf = vectorizer.fit_transform(x_train, )
X_test_tf = vectorizer.transform(x_test)
print(X_train_tf[0].shape)

### Belangrijkste woorden van TF IDF krijgen
feature_array = np.array(vectorizer.get_feature_names_out())
tfidf_sorting = np.argsort(X_test_tf.toarray()).flatten()[::-1]
n = 20
top_n = feature_array[tfidf_sorting][:n]
#print('belangrijkste woorden', top_n)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression


names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "Logistic Regression"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    MultinomialNB(),
    LogisticRegression(random_state=0)
]




for name, clf in zip(names, classifiers):


    classifier = clf 
    classifier.fit(X_train_tf, y_train)

    y_pred = classifier.predict(X_test_tf)

    score1 = metrics.accuracy_score(y_test, y_pred)

    print("Metrics for {}".format(name))
    print("accuracy:   %0.3f" % score1)

    print(metrics.classification_report(y_test, y_pred, target_names = ['Positive', 'Negative']))

    print("confusion matrix {}:".format(name))
    print(metrics.confusion_matrix(y_test, y_pred))

    print('------------------------------')

