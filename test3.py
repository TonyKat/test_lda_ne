# -*- coding: utf-8 -*-

from functools import reduce

items = [[1,2],[3,4]]
sum_all = reduce(lambda x, y: x + y, items)

print(sum_all)

import random
import collections
from collections import OrderedDict
from string import punctuation


c = collections.Counter()
print('Initial :', c)

ddd = {'d': 6, 'a': 4, 'b': 2, 'c': 1}

c.update('abcdaab')
print('Sequence:', c)

c.update({'a':1, 'd':5})
print('Dict    :', c)
print(ddd == c, 'Верно')
dic = {}
for item in ['asd','2','44fd', 'gfd', '2', '2']:
     if item in dic:
         dic[item] += 1
     else:
         dic[item] = 1

arr = dic.keys()
arrv= dic.values()
print(arr, arrv)
dic.update({'asd':14, 'FFFY':1})
print(dic)
dic.update({'FFFY':dic['FFFY'] + 1})
#print(dic.keys(), dic.values())
print(len(dic))
# Пустой словарь.
my_dict = dict()

# Словарь из итерирующегося объекта.
my_dict = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
print(my_dict)
# Словарь из именованных аргументов.
my_dict = dict(one=1, two=2, three=3)
print(my_dict)
# Словарь из списка кортежей.
my_dict = dict([('two', 2), ('one', 1), ('three', 3)])
print(my_dict)
my_dict_2 = OrderedDict(my_dict.items())
print([i for i in my_dict_2])
#print(my_dict_2[0],my_dict_2[1],my_dict_2[2])
mas = [[1,2],
       [3,4],
       [5,6]]
print(mas[0][0],mas[1][1], '---')
docs = ['asd', '123', 'lkj', 'fnk']
for i,doc in enumerate(docs):
    print(i,doc)
class ItemWord:
    def __init__(self, docID = None, wordID = None, topicID = None):
        self.docID = docID
        self.wordID = wordID
        self.topicID = topicID
a = ItemWord
a.docID = 1
a.wordID = 2
a.topicID = 3
print(a.docID)
rand = random.randint(0, 10)
print(rand)
d = {}
import numpy as np
n_k_t = np.zeros((3,5))
n_k_t[2][2] = 1

topics = 2
docs = np.array(("칼로리 레시피 서비스 식재료 먹거리",
                "도시락 건강식 다이어트 칼로리 레시피",
                "마케팅 다이어트 식재료 배송 칼로리",
                "여행 YOLO 혼술 휴가 연휴",
                "여행 예약 항공권 마케팅 연휴",
                "항공권 예약 호텔 다구간 서비스"))

words_full = []
words_uniq = []
doc_word = np.zeros((docs.shape[0]))
doc_words_size = np.zeros((docs.shape[0]))
print('doc_word =',doc_word)
print('doc_words_size =',doc_words_size)
a = 0
for doc in docs:
    doc_words = doc.split()
    words_full += doc_words
    doc_words_size[a] = len(doc_words)
    a += 1
words_full = np.array(words_full)
print ("words_full")
print (words_full)

words = np.array(list(set(words_full)))
words_uniq = np.unique(words_full)
words_uniq = np.reshape(words_uniq, (words_uniq.shape[0]))
print ("words_uniq")
print (words_uniq)

word_doc_topic = np.array(['keyword', 0, 0, 0])
a=0
for doc in docs:
    words = doc.split()
    for word in words:
        id_uniq = np.where(words_uniq == word)[0]
        to = random.randrange(0, topics)
        element = (word, a, to, id_uniq[0])
        word_doc_topic = np.vstack((word_doc_topic, element))
    a += 1
word_doc_topic = word_doc_topic[1:, :]
print ("word_doc_topic")
print (word_doc_topic)
print(word_doc_topic[:, 1])
print(word_doc_topic[:, 2])
print(np.count_nonzero((word_doc_topic[:, 1] == str(0)) & (word_doc_topic[:, 2] == str(0))))


#np.where(condition,x,y) - если выполняется условие, то х, если нет, то y
print(np.where([[True, False], [True, True]],
          [[1, 2], [3, 4]],
          [[9, 8], [7, 6]]))
print([xv if c else yv for (c,xv,yv) in zip([[True, False], [True, True]],
          [[1, 2], [3, 4]],
          [[9, 8], [7, 6]])])

date = ['2002 гг', '28.06.2006', '1917 г.', '1937 гг', '1920 году', '1917 г.', '14 июля 1937 г.', '1999 г.','1920 году']
print(date.index('1920 году',5))
asd = [1,2,3]
dsa = asd.copy()
print(dsa)

class ItemWord:
    def __init__(self, docID = None, wordID = None, topicID = None):
        self.docID = docID
        self.wordID = wordID
        self.topicID = topicID

class ItemTopic:
    def __init__(self, topicID = None, probability = None):
        self.topicID = topicID
        self.probability = probability


class CheckItemWord:
    def __init__(self, docID = None, wordID = None, topicID = None, check = False):
        self.docID = docID
        self.wordID = wordID
        self.topicID = topicID
        self.check = check

w = ItemWord
w.docID = 123
w.wordID = 32
w.topicID = 43
aSample = [w]

check_aSample = [CheckItemWord for x in range(len(aSample))]
print(check_aSample)
for i in range(len(aSample)):
    check_aSample[i].docID = aSample[i].docID
    check_aSample[i].wordID = aSample[i].wordID
    check_aSample[i].topicID = aSample[i].topicID
    check_aSample[i].check = True

print(check_aSample[0].docID,check_aSample[0].wordID,check_aSample[0].topicID,check_aSample[0].check)

docs =['123 kjhkh h gjhg jhg ','ads hghljg, kjhlk , kjlkh,', 'df', 'er', 'jn']
for i,doc in enumerate(docs):
    for j,word in enumerate(doc.split()):
        print('i = {}, j = {}, word = {}'.format(i,j,word))

from collections import namedtuple
MyStruct = namedtuple("MyStruct", "field1 field2 field3")

m = MyStruct("foo", "bar", "baz")

m = MyStruct(field1="foo", field2="bar", field3="baz")
print(m.field1,m.field2,m.field3)
aaaa = [m,m,m]
print(aaaa[0].field3)


class CheckItemWord:
    def __init__(self, docID = None, wordID = None, topicID = None, check = False):
        self.docID = docID
        self.wordID = wordID
        self.topicID = topicID
        self.check = check
d = [CheckItemWord() for x in range(10)]
print(d[0].check)


exit()
n = c.keys()
print(n)
#print(n.values())

aSample_1 = [1,2,3]
aSample = list()
aSample.append(1);aSample.append(2);aSample.append(3)
print(aSample_1, aSample, aSample == aSample_1)

mypunctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':',
                   ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
                   '—', '«', '»', '…', '–']

def rm_char(original_str, need2rm):
    ''' Remove charecters in "need2rm" from "original_str" '''
    return original_str.translate(str.maketrans('','',need2rm))

text = '''
    !.!!! Простите, еще несколько цитат из приговора. «…!!Отрицал существование
    Иисуса и пророка Мухаммеда», «наделял Иисуса Христа качествами
    ожившего мертвеца — зомби» [и] «качествами покемонов —
    представителей бестиария японской мифологии, тем самым совершил
    преступление, предусмотренное статьей 148 УК РФ антону Ирины…
    Какой чудестный день?
    '''
text_new = '''''
     Санкт-Петербургский государственный университет Санкт-Петербургский государственный университет Фукуяма Ф. Конец истории /У Вопр. философии. 1990. №3. С. 143. 4 Шугуров М. В. Конституцнонализация международных норм нрав человека и российское правосознание !! Общественные науки и современность. 2006. №2. С' 68. ' См.: От тоталитарных стереотипов к демократической культуре. М.. 1991. '' Конфликты в современной России: проблемы анализа и регулирования. М.. 2000. С 79. 7	Там же, С.80. 8	См.: Горшков М..К. Российское общество в >словия\ трансформации: мифы и реальность (социологический анализ). 1992 2002 гг. М.. 2003. ' Паншин ПК. Демократия в России: '''
a = rm_char(text_new, ''.join(mypunctuation))

print(a)

b = text_new.translate(str.maketrans('','', ''.join(mypunctuation)))
print(b)

b_split = b.split()
print(b_split)

fake_num = [41, 48, 53, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
print(fake_num)
fake_num = reduce(lambda x, y: x + y, fake_num)
print(fake_num)

