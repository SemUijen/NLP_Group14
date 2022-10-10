from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


data = pd.read_csv(r'C:\Users\Elske\NLP local\aapl_us_equities_news_prep_text_consol_text_html_inval_upper_char_lemmas_stopwords.csv')

vectorizer = TfidfVectorizer()

x_train, x_test, y_train, y_test = train_test_split(data['text'], data['target'])


# TF-IDF matrix voor train en test data
X_train_tf = vectorizer.fit_transform(x_train)
X_test_tf = vectorizer.transform(x_test)

# Naive Bayes Classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, y_train)

y_pred = naive_bayes_classifier.predict(X_test_tf)

# compute the performance measures
score1 = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(y_test, y_pred))
                                            #target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

print('------------------------------')

