from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
# bron: https://iq.opengenus.org/naive-bayes-on-tf-idf-vectorized-matrix/


data = pd.read_csv(r'C:\Users\Elske\NLP local\aapl_us_equities_news_prep_text_consol_text_html_inval_upper_char_lemmas_stopwords.csv')

#vectorizer = TfidfVectorizer() # Alles wordt negatief predict
vectorizer = CountVectorizer() # Mooiere verdeling, slechtere accuracy

x_train, x_test, y_train, y_test = train_test_split(data['text'], data['target'], random_state= 42)

print(sum(y_train), len(y_train))


# TF-IDF matrix voor train en test data
X_train_tf = vectorizer.fit_transform(x_train)
X_test_tf = vectorizer.transform(x_test)

### Belangrijkste woorden van TF IDF krijgen
# print('Woorden', vectorizer.get_feature_names_out())
feature_array = np.array(vectorizer.get_feature_names_out())
tfidf_sorting = np.argsort(X_test_tf.toarray()).flatten()[::-1]
n = 20
top_n = feature_array[tfidf_sorting][:n]
print('belangrijkste woorden', top_n)


# Naive Bayes Classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, y_train)

y_pred = naive_bayes_classifier.predict(X_test_tf)

# compute the performance measures
score1 = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(y_test, y_pred, target_names = ['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

print('------------------------------')

