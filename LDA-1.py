import psycopg2
import pymorphy2
import numpy as np
import random
# from numba import jit
from tqdm import tqdm
from time import time
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
# from gensim import corpora
from collections import OrderedDict

'''cimport cython
cimport numpy as np
import warnings
DTYPE = np.int
ctypedef np.int_t DTYPE_t'''


class ItemNe:
    def __init__(self, is_ne=False, _type_ne=None):
        self.is_ne = is_ne
        self._type_ne = _type_ne


class ItemWord:
    def __init__(self, docID=None, wordID=None, topicID=None, _ne=ItemNe):
        self.docID = docID
        self.wordID = wordID
        self.topicID = topicID
        self._ne = _ne


class ItemTopic:
    def __init__(self, topicID=None, probability=None):
        self.topicID = topicID
        self.probability = probability

# !!! dict быстрее ифов
what_is_number_type_ne = {
    'Адрес': 0,
    'Дата': 1,
    'Местоположение': 2,
    'Деньги': 3,
    'Имя': 4,
    'Организация': 5,
    'Персона': 6
}

# NEW
def del_ne(span_ne_loc, type_ne_loc, ne_loc, text_):
    re_span = np.array([[-1, -1] for x in range(len(span_ne_loc))])
    indx_span = np.array([], dtype=int)
    sum_ = 0
    for i in range(len(span_ne_loc)):
        if span_ne_loc[i] in re_span:
            indx_span = np.append(indx_span, i)
            continue
        if i != 0 and span_ne_loc[i][0] in range(span_ne_loc[i - 1][0], span_ne_loc[i - 1][1] + 1):
            indx_span = np.append(indx_span, i)
            continue
        re_span[i] = span_ne_loc[i]
        text_ = text_[0:span_ne_loc[i][0] - sum_] + text_[span_ne_loc[i][1] - sum_:len(text_)]
        sum_ += (span_ne_loc[i][1] - span_ne_loc[i][0])

    re_span = re_span[re_span != [-1, -1]]
    re_span = re_span.reshape(int(len(re_span) / 2), 2)
    ne_loc = np.delete(ne_loc, indx_span)
    type_ne_loc = np.delete(type_ne_loc, indx_span)

    return text_, ne_loc, re_span, type_ne_loc

# !!!
tokenizer = RegexpTokenizer(r'\w+')
ru_stop = get_stop_words('russian')
morph = pymorphy2.MorphAnalyzer()

# NEW
def normal_word(doc):
    # clean and tokenize document string
    raw_lower = doc.lower()
    tokens = tokenizer.tokenize(raw_lower)
    # remove stop words from tokens
    stopped_tokens = np.array([k for k in tokens if k not in ru_stop])
    # stem tokens (нормализация слов)
    stemmed_tokens = np.array([morph.normal_forms(l)[0] for l in stopped_tokens])
    # add tokens to list
    return stemmed_tokens


def test_tf_matrix():
    from skbayes.decomposition_models import GibbsLDA
    from sklearn.feature_extraction.text import CountVectorizer
    f_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                   stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    print(("Number of documents is {0}, vocabulary size is {1}").format(tf.shape[0], tf.shape[1]))
    return

# NEW ordered_dict_new = np.array([x for x in dict_words])
def tf_matrix_and_dict_new():
    t_tf = time()
    cur.execute('''SELECT count(*) FROM public.news_rbc''')
    len_db = cur.fetchone()[0]
    matrix_tf_2 = np.array([])
    dict_words = set()
    for i in tqdm(range(0, 6000), desc='tf_matrix_and_dict_new'):
        cur.execute('''SELECT full_text,ne,span_ne,type_ne FROM public.news_rbc WHERE id=%s''', (i,))
        row = cur.fetchone()
        if row[1] is None or row[1] == [] or row[1] == 0:
            continue
        text = row[0]
        ne = row[1]
        span_ne = row[2]
        type_ne = row[3]
        # удалить NE из текста и нормализовать его
        span_ne, ne, type_ne = (np.array(x) for x in zip(*sorted(zip(span_ne, ne, type_ne))))
        text, ne, span_ne, type_ne = del_ne(span_ne, type_ne, ne, text)
        text = normal_word(text)
        # запомнить номера NE после текста
        len_text_without_ne = len(text)
        num_ne = np.array([len_text_without_ne + x for x in range(len(ne))])
        # добавить в нормализованный текст NE
		# !!!
        text = np.append(text,ne)
        # -=-получить словарь-=-
        # разбить документ на слова по разделителю '|'
        split_doc = text
        # получить уникальные слова документа и их частоту
        unique_local, counts_local = np.unique(split_doc, return_counts=True)
        dict_with_count_local = dict(zip(unique_local, counts_local))
        matrix_tf_2 = np.append(matrix_tf_2, dict_with_count_local)
        # записать в словарь новые слова
        dict_words.update(unique_local)

        # записать в БД нормализованный текст и NE
        # преобразование типов, для БД (не работает с NumPy)
        ne = ne.tolist()
        span_ne = span_ne.tolist()
        num_ne = num_ne.tolist()
        type_ne = type_ne.tolist()
        '''cur.execute('INSERT INTO public.news_rbc_in_normal_form'
                    '(id, norm_text, ne, span_ne, num_ne_in_text, type_ne)'
                    'VALUES (%s, %s, %s, %s, %s, %s)',
                    (i, text, ne, span_ne, num_ne, type_ne))
        con.commit()'''
    ordered_dict_new = np.array([x for x in dict_words])
    print('Всего документов: {}'.format(len(matrix_tf_2)))
    print('Длина словаря = {}'.format(len(ordered_dict_new)))
    '''for i, word in enumerate(ordered_dict_new):
        cur.execute('INSERT INTO public.dictionary_news_rbc (id, word) VALUES (%s, %s)', (i, str(word)))
    con.commit()
    for i in tqdm(range(len(matrix_tf_2)), desc='tf_matrix in DB'):
        cur.execute('SELECT id FROM public.news_rbc_in_normal_form ORDER BY id ASC LIMIT 1 OFFSET %s',
                    (i,))
        doc_id = cur.fetchone()[0]
        for word_id in range(len(ordered_dict_new)):
            word = str(ordered_dict_new[word_id])
            tf = int(matrix_tf_2[i].get(word, 0))
            cur.execute('INSERT INTO public.matrix_tf (doc_id, word_id, tf, word) VALUES (%s, %s,%s, %s)',
                        (doc_id, word_id, tf, word))'''
    #con.commit()
    print('Время создания matrix_tf и ordered_dict: {}'.format(time() - t_tf))
    return ordered_dict_new


# NEW

def generative_process_new(dict_full, cur):
    t_generative = time()
    # получить кол-во элементов в БД = всего док-тов
    cur.execute('SELECT count(*) FROM public.news_rbc_in_normal_form')
    len_docs = cur.fetchone()[0]
    cur.execute('SELECT id, norm_text, ne, span_ne, num_ne_in_text, type_ne '
                'FROM public.news_rbc_in_normal_form ORDER BY id ASC '
                'LIMIT %s', (len_docs,))
    '''
    cur.execute('SELECT * FROM public.news_rbc_in_normal_form ORDER BY id ASC '
                'LIMIT %s', (len_docs,))
    '''
    rows = cur.fetchall()
    # количество слов в документе m, присвоенных теме k
    # number of words in document m that are assigned to the topic k
    n_m_k = np.zeros((len_docs, number_topic), dtype=int)

    # длина документа
    # document length
    n_m = np.zeros(len_docs, dtype=int)

    # количество совпадений термина t, присвоенного теме k в коллекции
    # number of occurences of term t assigned to the topic k in the collection
    n_k_t = np.zeros((number_topic, len(dict_full)), dtype=int)

    # number of terms assigned to the topic k
    # количество терминов присвоенных теме k
    n_k = np.zeros(number_topic, dtype=int)
    # ----------------------------------------

    # каждому слову - тему
    aSample_ = np.array([]) # !!! можно выделить нужное кол-во элементов и записывать по индексу, а не делать .append в {{**}}
    for i in tqdm(range(len_docs), desc='generative_process_new'):
        doc_id = rows[i][0]
        doc = rows[i][1]
        # ne = rows[i][2]
        # span_ne = rows[i][3]
        num_ne = rows[i][4]
        type_ne = rows[i][5]
        split_doc = doc.split('|')

        for j, word in enumerate(split_doc):
            word_id = np.where(dict_full == word)[0][0]
            w = ItemWord()
            w.docID = doc_id
            w.wordID = word_id
            w.topicID = random.randint(0, number_topic - 1)
            _is_ne = j in num_ne
            w._ne.is_ne = _is_ne
            w._ne._type_ne = what_is_number_type_ne[type_ne[num_ne.index(j)]] if _is_ne else None
            aSample_ = np.append(aSample_, w)  # !!! {{**}}

            m = w.docID
            t = w.wordID
            k = w.topicID

            n_m_k[m][k] += 1
            n_m[m] += 1
            n_k_t[k][t] += 1
            n_k[k] += 1
            # del row, doc_id, ...

    del rows
    print('Время в generative process: {}'.format(time() - t_generative))
    return aSample_, n_m_k, n_m, n_k_t, n_k


# NEW

def Gibbs_sampling_new(aSample, n_m_k, n_m, n_k_t, n_k, dict_full):
    time_gibbs = time()
    L = 500  # number of iterations
    len_dict_full = len(dict_full)
    for iter_L in tqdm(range(0, L), desc='Gibbs_sampling_new'):
        for wIndex, w in enumerate(aSample):
            m = w.docID
            t = w.wordID
            k_old = w.topicID

            # проверка на совпадение тем и NE
            if w._ne.is_ne == True and w._ne._type_ne == w.topicID:
                continue

            if n_m_k[m][k_old] == 0:
                n_m_k[m][k_old] = 0
            else:
                n_m_k[m][k_old] -= 1
            # если термин t в теме k_old встретился 0 раз, то = 0, иначе -= 1
            if n_k_t[k_old][t] == 0:
                n_k_t[k_old][t] = 0
            else:
                n_k_t[k_old][t] -= 1
            # если тема k_old в коллекции док-тов встретилась 0 раз, то = 0, иначе -= 1
            if n_k[k_old] == 0:
                n_k[k_old] = 0
            else:
                n_k[k_old] -= 1
            # длина док-та m -= 1
            if n_m[m] == 0:
                n_m[m] = 0
            else:
                n_m[m] -= 1

            # выбор новой темы
            aP = np.array([]) # !!! можно выделить нужное кол-во элементов и записывать по индексу, а не делать .append в {{*}}
            total = 0.0

            # вычисление по формуле
            # ОЧЕВИДНО!!!! ВОТ ЭТО НУЖНО ПОПРАВИТЬ И СДЕЛАТЬ ЧУТЬ ЛИ НЕ В ОДНУ СТРОЧКУ
            # пример_начало
            '''# retrieve doc, word and topic indices
               wi = words[i]
               di = docs[i]
               ti = topic_assignment[cum_i]

               # remove all 'influence' of i-th word in corpus
               word_topic[wi,ti] -= 1
               doc_topic[di,ti] -= 1
               topics[ti] -= 1

               # compute p(z_{n,d} = k| Z_{-n,d}) (i.e. probability of assigning
               # topic k for word n in document d, given all other topic assignments)
               p_z = (doc_topic[di] + alpha) / (alpha*n_topics + max(n_d[di,0] - 1,0) )

               # compute p(W|Z) (i.e. probability of observing corpus given all
               # topic assignments) and by multiplying it to p(z_{n,d} = k| Z_{-n,d})
               # obtain unnormalised p(z_{n,d}| DATA)
               p_z *= (word_topic[wi,:] + gamma) / (gamma*n_words + topics)

               # normalise & handle any conversion issues
               normalizer = np.sum(p_z)
               partial_sum = 0.0
               for k in xrange(self.n_topics-1):
                   p_z[k] /= normalizer
                   partial_sum += p_z[k]
               p_z[n_topics-1] = 1.0 - partial_sum

               # make sample from multinoulli distribution & update topic assignment
               ti = np.where(np.random.multinomial(1,p_z))[0][0]
               topic_assignment[cum_i] = ti

               # add 'influence' of i-th element in corpus back
               word_topic[wi,ti] += 1
               doc_topic[di,ti] += 1
               topics[ti] += 1
               cum_i += 1'''
            # пример_конец
            for k in range(0, number_topic):
                # если тема k встретилась в коллекции не 0 раз, то...
                sum_n_k_T = n_k[k] + beta * len_dict_full  # по формуле (79)
                # !!!
                sum_n_m_K = n_m[m] + alpha * number_topic  #
                # !!!
                # сколько раз термин t был упомянут в теме k
                n_k_t_K_T = n_k_t[k][t]
                # !!!
                # сколько слов в док-те m было присвоено теме k
                n_m_k_M_K = n_m_k[m][k]
                # !!!

                # p(k|t,(!k)) = вероятность того, что тема k, описывается термином t
                p = (n_k_t_K_T + beta) / sum_n_k_T * (n_m_k_M_K + alpha) / sum_n_m_K

                t_ = ItemTopic()
                t_.topicID = k
                t_.probability = p

                aP = np.append(aP, t_)  # !!! {{*}}

                # сумма условной вероятности: вероятность, что тема k описывается термином t
                total += p

            _sum = 0.0
            for t_ in aP:
                t_.probability = t_.probability / total
                _sum += t_.probability
            aP[number_topic - 1] = 1.0 - _sum
            k_new = np.where(np.random.multinomial(1, [t_.probability for t_ in aP]))[0][0]

            # обновить еще раз
            n_m_k[m][k_new] += 1
            n_k_t[k_new][t] += 1
            n_m[m] += 1
            n_k[k_new] += 1

            aSample[wIndex].topicID = k_new

        '''for w in aSample:
            cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id) '
                        'VALUES (%s, %s, %s, %s, %s)', (iter_L, w.docID, w.wordID, w.topicID))
        con.commit()'''

    # записать последнюю итерацию
    '''for w in aSample:
        cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id)'
                    ' VALUES (%s, %s, %s, %s, %s)', (L, w.docID, w.wordID, w.topicID))
    con.commit()'''

    print('Время в Gibbs_Sampling: {}'.format(time() - time_gibbs))
    return


if __name__ == '__main__':
    time_begin = time()
    # подключаемся к базе
    con = psycopg2.connect(database="texts_politics", user="postgres", password="197346qaz", host="127.0.0.1",
                           port="5432", )
    # создаем курсор для работы с базой данных
    cur = con.cursor()
    cur.execute('Select')
    number_topic = 100
    alpha = 50 / number_topic
    beta = 0.01

    ordered_dict = tf_matrix_and_dict_new()
    print('Время исполнения программы:', time() - time_begin)
    exit()
    cur.execute('SELECT count(*) FROM public.dictionary_news_rbc')
    len_dict = cur.fetchone()[0]
    cur.execute('SELECT word FROM public.dictionary_news_rbc ORDER BY id ASC LIMIT %s', (len_dict,))
    rows = cur.fetchall()
    ordered_dict = np.array([str(rows[x][0]) for x in range(len_dict)])
    '''docs = np.array([rows[x][1] for x in range(len_docs)])
    ne = np.array([rows[x][2] for x in range(len_docs)])
    span_ne = np.array([rows[x][3] for x in range(len_docs)])
    num_ne = np.array([rows[x][4] for x in range(len_docs)])
    type_ne = np.array([rows[x][5] for x in range(len_docs)])'''


    aSample, n_m_k, n_m, n_k_t, n_k = generative_process_new(ordered_dict, cur)
    Gibbs_sampling_new(aSample, n_m_k, n_m, n_k_t, n_k, ordered_dict)
    print('Время исполнения программы:', time() - time_begin)
    exit()
