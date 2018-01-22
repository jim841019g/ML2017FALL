# -*- coding: UTF-8 -*-
import string
import json
from gensim.models import Word2Vec,Doc2Vec
import jieba
import pickle
import sys
    
sentences = []
with open(sys.argv[1],'r') as f:
    train_data = json.load(f)    

context_list = []
question_list = []
id_list = []

for topic in train_data['data']:
    title = topic['title']
    for p in topic['paragraphs']:
        qas = p['qas'] 
        context = p['context']
        for qa in qas:
            context_temp = []
            for word in context:
                context_temp.append(word)
            context_list.append(context_temp)
            id = qa['id']
            id_list.append(id)
            questions = qa['question']
            questions_temp = []
            for word in questions:
                questions_temp.append(word)
            question_list.append(questions_temp)
with open('test_contexts', 'wb') as fp:
    pickle.dump(context_list, fp)
with open('test_questions', 'wb') as fp:
    pickle.dump(question_list, fp)
with open('test_id', 'wb') as fp:
    pickle.dump(id_list, fp,protocol=2)



