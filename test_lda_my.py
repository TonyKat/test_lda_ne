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

class ItemWordNew:
    def __init__(self, docID=None, wordID=None, topicID=None, type_ne_=None):
        self.docID = docID
        self.wordID = wordID
        self.topicID = topicID
        self.type_ne_ = type_ne_

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


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.word_topic_.T):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def print_top_words_new(word_topic, feature_names, n_top_words):
    for topic_idx, topic in enumerate(word_topic):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def new_data_set():
    cur.execute('SELECT "docID", "wordID", tf, type_ne FROM public.dataset_test ORDER BY "docID" ASC')
    rows = cur.fetchall()
    r = np.array(rows)
    a = r[:, 0]
    print(r[1],r[1,0],r[1][1],len(r),len(a))
    exit()
    fake = [0,1,2,3,5,6]
    dict_id = []
    for i in tqdm(range(len(rows)), desc='new_data_set'):
        if i == 0: continue
        cur.execute('SELECT count(*) FROM public.dataset_test WHERE "wordID"=%s',(rows[i][1],))
        count = cur.fetchone()[0]
        if rows[i][3] in fake or count < 2:
            continue
        cur.execute('INSERT INTO public.dataset_new("docID", "wordID", tf, type_ne)	VALUES (%s, %s, %s, %s)',
                    (rows[i][0],rows[i][1],rows[i][2],rows[i][3]))
        if rows[i][1] not in dict_id:
            dict_id.append(rows[i][1])
    con.commit()
    for i in dict_id:
        cur.execute('SELECT word, is_ne, type_ne FROM public.dictionary_news_rbc WHERE id=%s',(i,))
        row = cur.fetchone()
        cur.execute('INSERT INTO public.dictionary_new(id,word,is_ne,type_ne) VALUES (%s, %s, %s, %s)',
                    (i, row[0], row[1], row[2]))
    con.commit()
    return


def test_tf_matrix(data_samples):
    from skbayes.decomposition_models import GibbsLDA
    from sklearn.feature_extraction.text import CountVectorizer

    cur.execute('SELECT count(*) FROM public.news_rbc_in_normal_form')
    len_docs = cur.fetchone()[0]
    cur.execute('SELECT norm_text '
                'FROM public.news_rbc_in_normal_form ORDER BY id ASC '
                'LIMIT %s', (len_docs,))
    rows = cur.fetchall()
    data = [x[0] for x in rows]
    cur.execute('SELECT count(*) FROM public.dictionary_news_rbc')
    len_dict = cur.fetchone()[0]
    cur.execute('SELECT word FROM public.dictionary_news_rbc ORDER BY id ASC LIMIT %s', (len_dict,))
    rows = cur.fetchall()
    ordered_dict = [str(rows[x][0]) for x in range(len_dict)]
    print('Длина словаря (оригинального) = {}'.format(len_dict))
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000,
                                   stop_words=ru_stop, vocabulary=ordered_dict, lowercase=False)
    #print('tf_vectorizer = {}'.format(tf_vectorizer))
    t0 = time()
    tf = tf_vectorizer.fit_transform(data)
    #print(tf[0:10])
    print(tf[0])
    print(type(tf),tf.shape[0],tf.shape[1])
    print()
    a = tf[0].nonzero()
    print(a)
    row = a[0]
    col = a[1]
    print(row[0],col[0])
    #if col[0] in ordered_dict
    print('---')
    tf_feature_names = tf_vectorizer.get_feature_names()
    print(tf_feature_names[0:10])
    print(len(tf_feature_names))
    print(type(tf_feature_names))
    print("done in %0.3fs." % (time() - t0))
    print(("Number of documents is {0}, vocabulary size is {1}").format(tf.shape[0], tf.shape[1]))
    #lda = GibbsLDA(n_topics=100, n_burnin=500, verbose=True)
    #document_topic = lda.fit_transform(tf)
    print('Время:', time()-time_begin)

    #tf_feature_names = tf_vectorizer.get_feature_names()
    #print_top_words(lda, tf_feature_names, 20)
    return


def new_dataset():
    t_tf = time()
    # получить уже созданный словарь
    cur.execute('SELECT word,type_ne FROM public.dictionary_news_rbc ORDER BY id ASC;')
    words_and_type_ne = cur.fetchall()
    ordered_dict_new = np.array([x[0] for x in words_and_type_ne])
    type_ne_new = np.array([x[1] for x in words_and_type_ne])
    # пройтись по всем док-там и записать в БД # docID, wordID, tf, type_ne
    sum_ = 0
    idd = 0
    for i in tqdm(range(0, 6000), desc='in DB (docID,wordID,tf,type_ne)'):
        cur.execute('''SELECT full_text,ne,span_ne,type_ne FROM public.news_rbc WHERE id=%s''', (i,))
        row = cur.fetchone()
        if row[1] is None or row[1] == [] or row[1] == 0:
            sum_ += 1
            continue
        text = row[0]
        ne = row[1]
        span_ne = row[2]
        type_ne = row[3]
        # удалить NE из текста и нормализовать его
        span_ne, ne, type_ne = (np.array(x) for x in zip(*sorted(zip(span_ne, ne, type_ne))))
        text, ne, span_ne, type_ne = del_ne(span_ne, type_ne, ne, text)
        text = normal_word(text)
        # добавить в нормализованный текст NE
        text = np.append(text,ne)
        # получить уникальные слова документа и их частоту
        unique_local, counts_local = np.unique(text, return_counts=True)
        dict_with_count_local = dict(zip(unique_local, counts_local))
        '''cur.execute('UPDATE public.news_rbc_in_normal_form SET len_doc=%s WHERE id=%s',
                    (len(text), int(i)))'''
        for word, tf in dict_with_count_local.items():
            word_id = int(np.where(ordered_dict_new == word)[0][0])
            if type_ne_new[word_id] is not None:
                cur.execute('INSERT INTO public.dataset_new("docID", "wordID", tf, type_ne, id) '
                        'VALUES (%s, %s, %s, %s, %s)',(i - sum_, word_id, int(tf), type_ne_new[word_id], idd))
            else:
                cur.execute('INSERT INTO public.dataset_new("docID", "wordID", tf, type_ne, id) '
                            'VALUES (%s, %s, %s, %s, %s)', (i - sum_, word_id, int(tf), int(-1), idd))
            idd += 1
    con.commit()

    #print('Всего документов: {}'.format(5981))
    #print('Длина словаря = {}'.format(len(ordered_dict_new)))
    print('Время создания matrix_tf и ordered_dict: {}'.format(time() - t_tf))
    return


'''sum_new_tf_matrix = sum(sum(new_tf_matrix))
import math

tf_idf = np.zeros((new_tf_matrix.shape[0], new_tf_matrix.shape[1]))
for i in range(new_tf_matrix.shape[0]):
    for j in range(new_tf_matrix.shape[1]):
        if float(sum(new_tf_matrix[i])) != 0:
            tf = float(new_tf_matrix[i][j]) / float(sum(new_tf_matrix[i]))
        else:
            tf = 0.0
        idf = math.log10(float(new_tf_matrix.shape[0]) / float(np.count_nonzero(new_tf_matrix.T[j], axis=0)))
        tf_idf[i][j] = tf * idf'''

# new_2_begin
def generative_process_new_2(cur):
    t_generative = time()

    # получить длину уже созданного словаря
    cur.execute('SELECT count(*) FROM public.dictionary_news_rbc')
    len_dict = cur.fetchone()[0]

    # получить кол-во элементов в БД = всего док-тов
    cur.execute('SELECT count(*) FROM public.news_rbc_in_normal_form')
    len_docs = cur.fetchone()[0]
    cur.execute('SELECT "docID", "wordID", tf, type_ne FROM public.dataset_test ORDER BY "docID" ASC')
    rows = np.array(cur.fetchall())

    rows = np.array([row for row in rows if (row[3] == -1 or row[3] == 4) and row[2] < 5])

    # количество слов в документе m, присвоенных теме k
    # number of words in document m that are assigned to the topic k
    n_m_k = np.zeros((len_docs, number_topic), dtype=int)  # doc_topic[di,ti]

    # длина документа
    # document length
    n_m = np.zeros(len_docs, dtype=int)  # n_d[di,0]

    # количество совпадений термина t, присвоенного теме k в коллекции
    # number of occurences of term t assigned to the topic k in the collection
    n_k_t = np.zeros((number_topic, len_dict), dtype=int)  # word_topic[wi,ti]

    # number of terms assigned to the topic k
    # количество терминов присвоенных теме k
    n_k = np.zeros(number_topic, dtype=int)  # topics[ti]
    # ----------------------------------------

    # каждому слову - тему
    cur.execute('SELECT count(*) FROM public.dataset_test')
    len_words = cur.fetchone()[0]
    aSample_ = np.arange(len_words)
    for i in tqdm(range(len_words), desc='generative_process_new_2'):
        #doc_id_ = rows[i][0]  # docID
        #word_id_ = rows[i][1]  # wordID
        #tf_word_ = rows[i][2]  # tf
        #type_ne_word_ = rows[i][3]  # type_ne = [0:6] or -1'''

        for j in range(rows[i][2]):
            w = ItemWordNew(docID=rows[i][0],
                            wordID=rows[i][1],
                            topicID=random.randint(0, number_topic - 1),
                            type_ne_=rows[i][3])
            #w.docID = doc_id_
            #w.wordID = word_id_
            #w.topicID = random.randint(0, number_topic - 1)
            #w.type_ne_ = type_ne_word_

            aSample_ = np.append(aSample_, w)

            #m = w.docID
            #t = w.wordID
            #k = w.topicID

            n_m_k[w.docID][w.topicID] += 1
            n_m[w.docID] += 1
            n_k_t[w.topicID][w.wordID] += 1
            n_k[w.topicID] += 1


    del rows
    print('Время в generative process: {}'.format(time() - t_generative))
    return aSample_, n_m_k, n_m, n_k_t, n_k, len_dict


def Gibbs_sampling_new_2(aSample, n_m_k, n_m, n_k_t, n_k, len_dict):
    time_gibbs = time()
    L = 500  # number of iterations

    # new_gibbs_begin
    for iter_L in tqdm(range(0, L), desc='Gibbs_Sampling'):
        for i in range(len(aSample)):
            '''
            # количество слов в документе m, присвоенных теме k
            # number of words in document m that are assigned to the topic k
            n_m_k = np.zeros((len_docs, number_topic), dtype=int)  # doc_topic[di,ti]

            # длина документа
            # document length
            n_m = np.zeros(len_docs, dtype=int)  # n_d[di,0]

            # количество совпадений термина t, присвоенного теме k в коллекции
            # number of occurences of term t assigned to the topic k in the collection
            n_k_t = np.zeros((number_topic, len(dict_full)), dtype=int)  # word_topic[wi,ti]

            # number of terms assigned to the topic k
            # количество терминов присвоенных теме k
            n_k = np.zeros(number_topic, dtype=int)  # topics[ti]
            '''
            # retrieve doc, word and topic indices
            wi = aSample[i].wordID  # words[i]
            di = aSample[i].docID  # docs[i]
            ti = aSample[i].topicID  # topic_assignment[cum_i]
            type_ne_wi = aSample[i].type_ne_

            # remove all 'influence' of i-th word in corpus
            n_k_t[wi, ti] -= 1  # word_topic[wi,ti]
            n_m_k[di, ti] -= 1  # doc_topic[di,ti]
            n_k[ti] -= 1  # topics[ti]
            n_m[di] -= 1  # n_d[di,0] ---!!!---ИЗМЕНИЛ---!!!---

            # compute p(z_{n,d} = k| Z_{-n,d}) (i.e. probability of assigning
            # topic k for word n in document d, given all other topic assignments)
            p_z = (n_m_k[di] + alpha) / (alpha * number_topic + max(n_m[di], 0))  # ---!!!---ИЗМЕНИЛ---!!!---

            # compute p(W|Z) (i.e. probability of observing corpus given all
            # topic assignments) and by multiplying it to p(z_{n,d} = k| Z_{-n,d})
            # obtain unnormalised p(z_{n,d}| DATA)
            p_z *= (n_k_t[wi, :] + beta) / (beta * len_dict + n_k)

            # normalise & handle any conversion issues
            normalizer = np.sum(p_z)
            partial_sum = 0.0
            for k in range(number_topic - 1):
                p_z[k] /= normalizer
                partial_sum += p_z[k]
            p_z[number_topic - 1] = 1.0 - partial_sum

            # make sample from multinoulli distribution & update topic assignment
            ti_new = int(np.where(np.random.multinomial(1, p_z))[0][0])

            # проверка на совпадение тем
            if type_ne_wi != -1 and type_ne_wi == ti:
                n_k_t[wi, ti] += 1
                n_m_k[di, ti] += 1
                n_k[ti] += 1
                n_m[di] += 1
                continue
            elif type_ne_wi != -1 and type_ne_wi == ti_new:
                n_k_t[wi, ti_new] += 1
                n_m_k[di, ti_new] += 1
                n_k[ti_new] += 1
                n_m[di] += 1
                aSample[i].topicID = ti_new
                continue
            else:
                # add 'influence' of i-th element in corpus back
                n_k_t[wi, ti] += 1
                n_m_k[di, ti] += 1
                n_k[ti] += 1
                n_m[di] += 1
                aSample[i].topicID = ti_new
            '''for w in aSample:
                cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id) '
                                    'VALUES (%s, %s, %s, %s, %s)', (iter_L, w.docID, w.wordID, w.topicID))
            con.commit()'''
            # записать последнюю итерацию
    for w in aSample:
        b = (w.topicID == w.type_ne_)
        cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id, type_ne, is_true)'
                    ' VALUES (%s, %s, %s, %s, %s, %s)', (int(L), int(w.docID), int(w.wordID), int(w.topicID),
                                                         int(w.type_ne_), b))
    con.commit()

    # new_gibbs_end
    print('Время в Gibbs_Sampling: {}'.format(time() - time_gibbs))

    empty_docs = n_m[:] == 0
    dtd = np.array(n_m_k, dtype=np.double)
    dtd[empty_docs, :] = 1.0 / number_topic
    dtd[~empty_docs, :] = dtd[~empty_docs, :] / n_m[~empty_docs]
    print('Время в Gibbs_Sampling: {}'.format(time() - time_gibbs))

    cur.execute('INSERT INTO public.newsrbc(id,str) VALUES (%s, %s)', (int(0), n_k_t.tolist()))
    cur.execute('INSERT INTO public.newsrbc(id,str) VALUES (%s, %s)', (int(1), dtd.tolist()))
    con.commit()

    return dtd, n_k_t

# new_2_end


# new_3_begin
def generative_process_new_3(cur):
    t_generative = time()

    # получить длину уже созданного словаря
    cur.execute('SELECT count(*) FROM public.dictionary_news_rbc')
    len_dict = cur.fetchone()[0]

    # получить кол-во элементов в БД = всего док-тов
    cur.execute('SELECT count(*) FROM public.news_rbc_in_normal_form')
    len_docs = cur.fetchone()[0]
    cur.execute('SELECT "docID", "wordID", tf, type_ne, "topicID" FROM public.dataset_test ORDER BY "docID" ASC')
    rows = np.array(cur.fetchall())

    #rows = np.array([row for row in rows if (row[3] == -1 or row[3] == 4) and row[2] < 5])

    # количество слов в документе m, присвоенных теме k
    # number of words in document m that are assigned to the topic k
    n_m_k = np.zeros((len_docs, number_topic), dtype=int)  # doc_topic[di,ti]

    # длина документа
    # document length
    n_m = np.zeros(len_docs, dtype=int)  # n_d[di,0]

    # количество совпадений термина t, присвоенного теме k в коллекции
    # number of occurences of term t assigned to the topic k in the collection
    n_k_t = np.zeros((len_dict, number_topic), dtype=int)  # word_topic[wi,ti]:ОТЛИЧИЕ размерность(темыXслова) [уже нет]

    # number of terms assigned to the topic k
    # количество терминов присвоенных теме k
    n_k = np.zeros(number_topic, dtype=int)  # topics[ti]
    # ----------------------------------------
    print('words in corpus docs =', sum(rows[:, 2]))
    topic_assignment = np.random.randint(0, number_topic, sum(rows[:, 2]), dtype=np.int)
    corpus_id = 0
    # каждому слову - тему
    for i in tqdm(range(len(rows)), desc='generative_process_new_3'):
        #doc_id_ = rows[i][0]  # docID
        #word_id_ = rows[i][1]  # wordID
        #tf_word_ = rows[i][2]  # tf
        #type_ne_word_ = rows[i][3]  # type_ne = [0:6] or -1'''
        #topic_id = rows[i][4]  # topicID

        for j in range(rows[i][2]):
            rows[i][4] = topic_assignment[corpus_id]

            n_m_k[rows[i][0]][rows[i][4]] += 1
            n_m[rows[i][0]] += 1
            n_k_t[rows[i][1]][rows[i][4]] += 1
            n_k[rows[i][4]] += 1

            corpus_id += 1


    #del rows

    print('Время в generative process: {}'.format(time() - t_generative))
    return rows, n_m_k, n_m, n_k_t, n_k, len_dict, topic_assignment


def Gibbs_sampling_new_3(rows, n_m_k, n_m, n_k_t, n_k, len_dict, topic_assignment):
    time_gibbs = time()
    L = 500  # number of iterations

    # new_gibbs_begin
    for iter_L in tqdm(range(0, L), desc='Gibbs_Sampling_new_3'):
        cum_i = 0
        for i in range(len(rows)):
            # doc_id_ = rows[i][0]  # docID
            # word_id_ = rows[i][1]  # wordID
            # tf_word_ = rows[i][2]  # tf
            # type_ne_word_ = rows[i][3]  # type_ne = [0:6] or -1'''
            # topic_id = rows[i][4]  # topicID
            for j in range(rows[i][2]):
                # retrieve doc, word and topic indices
                #wi = rows[i][1]  # words[i]
                #di = rows[i][0]  # docs[i]
                #ti = topic_assignment[cum_i]  # topic_assignment[cum_i]
                #type_ne_wi = rows[i][3]

                # remove all 'influence' of i-th word in corpus
                n_k_t[rows[i][1], topic_assignment[cum_i]] -= 1  # word_topic[wi,ti]
                n_m_k[rows[i][0], topic_assignment[cum_i]] -= 1  # doc_topic[di,ti]
                n_k[topic_assignment[cum_i]] -= 1  # topics[ti]
                n_m[rows[i][0]] -= 1  # n_d[di,0] ---!!!---ИЗМЕНИЛ---!!!---

                # compute p(z_{n,d} = k| Z_{-n,d}) (i.e. probability of assigning
                # topic k for word n in document d, given all other topic assignments)
                p_z = (n_m_k[rows[i][0]] + alpha) / (alpha * number_topic + max(n_m[rows[i][0]], 0))

                # compute p(W|Z) (i.e. probability of observing corpus given all
                # topic assignments) and by multiplying it to p(z_{n,d} = k| Z_{-n,d})
                # obtain unnormalised p(z_{n,d}| DATA)
                p_z *= (n_k_t[rows[i][1], :] + beta) / (beta * len_dict + n_k)

                # normalise & handle any conversion issues
                normalizer = np.sum(p_z)
                partial_sum = 0.0
                for k in range(number_topic - 1):
                    p_z[k] /= normalizer
                    partial_sum += p_z[k]
                p_z[number_topic - 1] = 1.0 - partial_sum

                # make sample from multinoulli distribution & update topic assignment
                ti_new = int(np.where(np.random.multinomial(1, p_z))[0][0])

                # проверка на совпадение тем
                if rows[i][3] != -1 and rows[i][3] == topic_assignment[cum_i]:
                    ti_new = topic_assignment[cum_i]

                topic_assignment[cum_i] = ti_new

                n_k_t[rows[i][1], ti_new] += 1
                n_m_k[rows[i][0], ti_new] += 1
                n_k[ti_new] += 1
                n_m[rows[i][0]] += 1
                rows[i][3] = ti_new

                cum_i += 1

                '''for w in aSample:
                    cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id) '
                                        'VALUES (%s, %s, %s, %s, %s)', (iter_L, w.docID, w.wordID, w.topicID))
                con.commit()'''
                # записать последнюю итерацию
    rows = rows.tolist()
    for row in rows:
        # doc_id_ = rows[i][0]  # docID
        # word_id_ = rows[i][1]  # wordID
        # tf_word_ = rows[i][2]  # tf
        # type_ne_word_ = rows[i][3]  # type_ne = [0:6] or -1'''
        # topic_id = rows[i][4]  # topicID
        b = (row[4] == row[3])
        cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id, type_ne, is_true)'
                    ' VALUES (%s, %s, %s, %s, %s, %s)', (int(L), row[0], row[1], row[4],
                                                         row[3], b))
    con.commit()

    # new_gibbs_end
    #print('Время в Gibbs_Sampling: {}'.format(time() - time_gibbs))

    empty_docs = n_m[:] == 0
    dtd = np.array(n_m_k, dtype=np.double)
    dtd[empty_docs, :] = 1.0 / number_topic
    dtd[~empty_docs, :] = dtd[~empty_docs, :] / n_m[~empty_docs]
    print('Время в Gibbs_Sampling: {}'.format(time() - time_gibbs))

    cur.execute('INSERT INTO public.newsrbc(id,str) VALUES (%s, %s)', (int(0), n_k_t.tolist()))
    cur.execute('INSERT INTO public.newsrbc(id,str) VALUES (%s, %s)', (int(1), dtd.tolist()))
    con.commit()

    return dtd, n_k_t


def Gibbs_sampling_new_3_original(rows, n_m_k, n_m, n_k_t, n_k, len_dict, topic_assignment):
    time_gibbs = time()
    L = 500  # number of iterations

    # new_gibbs_begin
    for iter_L in tqdm(range(0, L), desc='Gibbs_Sampling_new_3'):
        cum_i = 0
        for i in range(len(rows)):
            # doc_id_ = rows[i][0]  # docID
            # word_id_ = rows[i][1]  # wordID
            # tf_word_ = rows[i][2]  # tf
            # type_ne_word_ = rows[i][3]  # type_ne = [0:6] or -1'''
            # topic_id = rows[i][4]  # topicID
            for j in range(rows[i][2]):
                # retrieve doc, word and topic indices
                wi = rows[i][1]  # words[i]
                di = rows[i][0]  # docs[i]
                ti = topic_assignment[cum_i]  # topic_assignment[cum_i]
                type_ne_wi = rows[i][3]

                # remove all 'influence' of i-th word in corpus
                n_k_t[wi, ti] -= 1  # word_topic[wi,ti]
                n_m_k[di, ti] -= 1  # doc_topic[di,ti]
                n_k[ti] -= 1  # topics[ti]
                n_m[di] -= 1  # n_d[di,0] ---!!!---ИЗМЕНИЛ---!!!---

                # compute p(z_{n,d} = k| Z_{-n,d}) (i.e. probability of assigning
                # topic k for word n in document d, given all other topic assignments)
                p_z = (n_m_k[di] + alpha) / (alpha * number_topic + max(n_m[di], 0))  # ---!!!---ИЗМЕНИЛ---!!!---

                # compute p(W|Z) (i.e. probability of observing corpus given all
                # topic assignments) and by multiplying it to p(z_{n,d} = k| Z_{-n,d})
                # obtain unnormalised p(z_{n,d}| DATA)
                p_z *= (n_k_t[wi, :] + beta) / (beta * len_dict + n_k)

                # normalise & handle any conversion issues
                normalizer = np.sum(p_z)
                partial_sum = 0.0
                for k in range(number_topic - 1):
                    p_z[k] /= normalizer
                    partial_sum += p_z[k]
                p_z[number_topic - 1] = 1.0 - partial_sum

                # make sample from multinoulli distribution & update topic assignment
                ti_new = int(np.where(np.random.multinomial(1, p_z))[0][0])

                # проверка на совпадение тем
                if type_ne_wi != -1 and type_ne_wi == ti:
                    ti_new = ti

                topic_assignment[cum_i] = ti_new

                n_k_t[wi, ti_new] += 1
                n_m_k[di, ti_new] += 1
                n_k[ti_new] += 1
                n_m[di] += 1
                rows[i][3] = ti_new

                cum_i += 1

                '''for w in aSample:
                    cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id) '
                                        'VALUES (%s, %s, %s, %s, %s)', (iter_L, w.docID, w.wordID, w.topicID))
                con.commit()'''
                # записать последнюю итерацию
    rows = rows.tolist()
    for row in rows:
        # doc_id_ = rows[i][0]  # docID
        # word_id_ = rows[i][1]  # wordID
        # tf_word_ = rows[i][2]  # tf
        # type_ne_word_ = rows[i][3]  # type_ne = [0:6] or -1'''
        # topic_id = rows[i][4]  # topicID
        b = (row[4] == row[3])
        cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id, type_ne, is_true)'
                    ' VALUES (%s, %s, %s, %s, %s, %s)', (int(L), row[0], row[1], row[4],
                                                         row[3], b))
    con.commit()

    # new_gibbs_end
    #print('Время в Gibbs_Sampling: {}'.format(time() - time_gibbs))

    empty_docs = n_m[:] == 0
    dtd = np.array(n_m_k, dtype=np.double)
    dtd[empty_docs, :] = 1.0 / number_topic
    dtd[~empty_docs, :] = dtd[~empty_docs, :] / n_m[~empty_docs]
    print('Время в Gibbs_Sampling: {}'.format(time() - time_gibbs))

    cur.execute('INSERT INTO public.newsrbc(id,str) VALUES (%s, %s)', (int(0), n_k_t.tolist()))
    cur.execute('INSERT INTO public.newsrbc(id,str) VALUES (%s, %s)', (int(1), dtd.tolist()))
    con.commit()

    return dtd, n_k_t
# new_3_end

# NEW
def tf_matrix_and_dict_new():
    t_tf = time()
    cur.execute('''SELECT count(*) FROM public.news_rbc''')
    len_db = cur.fetchone()[0]
    matrix_tf_2 = np.array([])
    dict_words = set()
    dict_ne = {}
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

        # словарь ne-type_ne
        new_type_ne = [what_is_number_type_ne[x] for x in type_ne]
        upd_dict_ne = [{ne[x]: new_type_ne[x]} for x in range(len(new_type_ne))]
        [dict_ne.update(x) for x in upd_dict_ne]
        # записать в БД нормализованный текст и NE
        # преобразование типов, для БД (не работает с NumPy)
        #ne = ne.tolist()
        #span_ne = span_ne.tolist()
        #num_ne = num_ne.tolist()
        #type_ne = type_ne.tolist()
        '''cur.execute('INSERT INTO public.news_rbc_in_normal_form'
                    '(id, norm_text, ne, span_ne, num_ne_in_text, type_ne)'
                    'VALUES (%s, %s, %s, %s, %s, %s)',
                    (i, text, ne, span_ne, num_ne, type_ne))
        con.commit()'''
    ordered_dict_new = np.array([x for x in dict_words])
    print('_____________', ordered_dict_new[0:10])
    cur.execute('SELECT word FROM public.dictionary_news_rbc ORDER BY id ASC;')
    words = cur.fetchall()
    for i in range(len(ordered_dict_new)):
        ordered_dict_new[i] = words[i][0]
    print('_____________', ordered_dict_new[0:10])
    print('Всего документов: {}'.format(len(matrix_tf_2)))
    print('Длина словаря = {}'.format(len(ordered_dict_new)))
    for word_ne,num_type_ne in dict_ne.items():
        word_id = np.where(ordered_dict_new == word_ne)[0][0]
        cur.execute('UPDATE public.dictionary_news_rbc SET is_ne=%s, type_ne=%s WHERE id=%s',
                    (True, num_type_ne,int(word_id)))
    con.commit()
    exit()
    '''new_tf_matrix = np.zeros((matrix_tf_2.shape[0],len(ordered_dict_new)), dtype=int)
    for i in range(matrix_tf_2.shape[0]):
        for j in range(len(ordered_dict_new)):
            new_tf_matrix[i][j] = int(matrix_tf_2[i].get(ordered_dict[j], 0))

    sum_new_tf_matrix = sum(sum(new_tf_matrix))
    import math
    tf_idf = np.zeros((new_tf_matrix.shape[0], new_tf_matrix.shape[1]))
    for i in range(new_tf_matrix.shape[0]):
        for j in range(new_tf_matrix.shape[1]):
            if float(sum(new_tf_matrix[i])) != 0:
                tf = float(new_tf_matrix[i][j]) / float(sum(new_tf_matrix[i]))
            else:
                tf = 0.0
            idf = math.log10(float(new_tf_matrix.shape[0]) / float(np.count_nonzero(new_tf_matrix.T[j], axis=0)))
            tf_idf[i][j] = tf * idf

    cur.execute('INSERT INTO public.newsrbc (tf_idf) VALUES (%s) WHERE id=%s',(tf_idf,int(1)))
    con.commit()'''
    #cur.execute('''SELECT full_text,ne,span_ne,type_ne FROM public.news_rbc WHERE id=%s''', (i,))

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

    new_dataset()
    exit()
    #new_data_set()

    number_topic = 100
    alpha = 50 / number_topic
    beta = 0.01

    rows, n_m_k, n_m, n_k_t, n_k, len_dict, topic_assignment = generative_process_new_3(cur)
    dtd, n_k_t = Gibbs_sampling_new_3(rows, n_m_k, n_m, n_k_t, n_k, len_dict, topic_assignment)
    # получить уже созданный словарь
    cur.execute('SELECT word FROM public.dictionary_news_rbc ORDER BY id ASC;')
    words = cur.fetchall()
    ordered_dict_new = np.array([x[0] for x in words])

    print_top_words_new(n_k_t.T, ordered_dict_new, 20)
    print('Время исполнения программы:', time() - time_begin)
    exit()

    #aSample_, n_m_k, n_m, n_k_t, n_k, len_dict = generative_process_new_2(cur)
    #dtd, n_k_t = Gibbs_sampling_new_2(aSample_, n_m_k, n_m, n_k_t, n_k, len_dict)


    #test_tf_matrix(['a'])
    #ordered_dict = tf_matrix_and_dict_new()
    #new_dataset()
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
