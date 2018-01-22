# -*- coding: utf-8 -*
import string
import json
from gensim.models import Word2Vec,Doc2Vec
import gensim
import pickle
import numpy as np
    
context_list = pickle.load(open('test_contexts','rb'))
question_list = pickle.load(open('test_questions','rb'))

print ('loading....')
model = Word2Vec.load("word2vector.model")
print ('success load')
context_data = []

s_start = []
s_end = []
for i in range(len(context_list)):
    max_hit = 0
    max_start = 0
    max_end = 0
    period_list = []
    period_list.append(0)
    for j in range(len(context_list[i])):
        word = context_list[i][j]
        if word == "。":
            period_list.append(j)
    for j in range(len(period_list)-1):
        start = period_list[j]
        end = period_list[j+1]
        hit = 0
        for k in range(len(question_list[i])):
            for l in range(start,end):
                if context_list[i][l] == question_list[i][k]:
                    hit+=1
                    break
        if hit > max_hit:
            max_hit = hit
            max_start = start
            max_end = end
    s_start.append(max_start)
    s_end.append(max_end)
with open('test_start', 'wb') as fp:
    pickle.dump(s_start, fp,protocol=2)
with open('test_end', 'wb') as fp:
    pickle.dump(s_end, fp,protocol=2)

for i in range(len(context_list)):
    sentence = []
    for j in range(s_start[i],s_end[i]):
        word = context_list[i][j]
        if word in model.wv.vocab:
            word_vec = model.wv[word]
        else:
            word_vec = np.zeros((64))
        word_vec = np.array(word_vec)
        sentence.append(word_vec)
    sentence = np.array(sentence)
    context_data.append(sentence)

question_data = []
for i in range(len(question_list)):
    sentence = []
    for j in range(len(question_list[i])):
        word = question_list[i][j]
        if word in model.wv.vocab:
            word_vec = model.wv[word]
        else:
            word_vec = np.zeros((64))
        word_vec = np.array(word_vec)
        sentence.append(word_vec)
    sentence = np.array(sentence)
    question_data.append(sentence)
data_0 = []
data_0.append(context_data)
data_0.append(question_data)


data = []
data.append(data_0)
with open('test_data', 'wb') as fp:
    pickle.dump(data, fp,protocol=2)
