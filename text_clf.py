# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import numpy as np
import MeCab
import os

"""
in : '関数型言語では関数を第一級オブジェクトとして扱う。'
out: ['関数', '型', '言語', 'で', 'は', '関数', 'を', '第', '一', '級', 'オブジェクト', 'として', '扱う', '。']
"""
def make_noun_list(text) :
    tagger = MeCab.Tagger("mecabrc")
    noun_list = filter(lambda n : "名詞" in n, tagger.parse(text).split("\n"))
    return map(lambda n : n.split("\t")[0], noun_list)

def make_text_list(directory) :
    return map(lambda n : open(os.path.join(directory, n)).read(), os.listdir(directory))

learn_text_list = make_text_list("learn")
test_text_list = make_text_list("test")

vectorizer = TfidfVectorizer(analyzer=make_noun_list, min_df=1)

X = vectorizer.fit_transform(learn_text_list)
_X = vectorizer.transform(test_text_list)
y = np.array(["functional", "OOP"])

clf = MultinomialNB()
#clf = svm.NuSVC()
#clf = GaussianNB()
clf.fit(X.todense(), y)

print clf.predict(_X.todense())
