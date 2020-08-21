# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


news = fetch_20newsgroups(subset='all')
X_train,  X_test, y_train, y_test = train_test_split(news.data, news.target,\
                            test_size = 0.25, random_state = 33)

tfidf = TfidfVectorizer()

X_tfidf_train = tfidf.fit_transform(X_train)
X_tfidf_test = tfidf.transform(X_test)

mnb_count = MultinomialNB()
mnb_count.fit(X_tfidf_train, y_train)

print('The accuracy of the NB with TfidfVectorizer:', \
      mnb_count.score(X_tfidf_test, y_test))

y_predict = mnb_count.predict(X_tfidf_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=news.target_names))