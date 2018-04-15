import psycopg2
import pymorphy2
import numpy as np
import random
from tqdm import tqdm
from time import time
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
# from gensim import corpora
from collections import OrderedDict


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


def what_is_number_type_ne(type_ne_):
    if type_ne_ == 'Адрес':
        return 0
    if type_ne_ == 'Дата':
        return 1
    if type_ne_ == 'Местоположение':
        return 2
    if type_ne_ == 'Деньги':
        return 3
    if type_ne_ == 'Имя':
        return 4
    if type_ne_ == 'Организация':
        return 5
    if type_ne_ == 'Персона':
        return 6


def check_topic(word, ne, k, checker, index_w, type_ne):
    flag = 0
    if checker[index_w].check == True:
        flag = 1
        return flag, checker
    if word in ne:
        num_word_in_ne = ne.index(word)
        type_word_ne = type_ne[num_word_in_ne]  # поправить
        num_topic = what_is_number_type_ne(type_word_ne)
        if k == num_topic:
            checker[index_w].check = True
            flag = 1
    return flag, checker


def read_files(cur):
    data_text = []
    data_ne = []
    type_ne = []
    fake_num = [40, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    # [41, 48, 53, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
    for number_file in range(100):
        if number_file in fake_num:
            number_file += 1
            continue
        cur.execute("SELECT text,named_entities,type_ne FROM public.news_of_politics ORDER BY id ASC LIMIT 1 OFFSET %s",
                    (number_file,))
        text_file = cur.fetchone()
        data_text.append(text_file[0])
        data_ne.append(text_file[1])
        type_ne.append(text_file[2])
    '''
    topics:
    1 - Адрес
    2 - Дата
    3 - Местоположение
    4 - Деньги
    5 - Имя
    6 - Организация
    7 - Персона
    8 - no_name_1
    9 - no_name_2
    10 - no_name_3
    '''
    return data_text, data_ne, type_ne


def normal_form_word(_docs):
    tokenizer = RegexpTokenizer(r'\w+')
    ru_stop = get_stop_words('russian')
    morph = pymorphy2.MorphAnalyzer()
    texts = []
    for i in _docs:
        # clean and tokenize document string
        raw_lower = i.lower()
        tokens = tokenizer.tokenize(raw_lower)
        # remove stop words from tokens
        stopped_tokens = [k for k in tokens if k not in ru_stop]
        # stem tokens (нормализация слов)
        stemmed_tokens = [morph.normal_forms(l)[0] for l in stopped_tokens]
        # add tokens to list
        texts.append(' '.join(stemmed_tokens))
    return texts


def get_dict(docs):
    words = {}
    for doc in docs:
        for word in doc.split():
            if word in words:
                words.update({word: words[word] + 1})
            else:
                words.update({word: 1})
    number_words = 0
    for word in words:
        number_words += words[word]
    return words, number_words


def indexing_words_of_dict(words):
    dict_of_words = OrderedDict(words.items())
    ordered_dict = []
    for key in dict_of_words:
        ordered_dict.append(key)
    indx = np.arange(0, len(dict_of_words))
    return ordered_dict, indx


def tf_matrix(docs, dict):
    matrix_tf_word_in_docs = np.zeros((len(docs), len(dict)), dtype=int)
    for i, doc in enumerate(docs):
        local_dict = {}
        for word in doc.split():
            if word in local_dict:
                local_dict.update({word: local_dict[word] + 1})
            else:
                local_dict.update({word: 1})
        for j in range(len(dict)):
            if dict[j] in local_dict:
                matrix_tf_word_in_docs[i][j] = local_dict[dict[j]]
            else:
                matrix_tf_word_in_docs[i][j] = 0

    return matrix_tf_word_in_docs


def generative_process(matrix_tf, docs, dict):
    # количество слов в документе m, присвоенных теме k
    # number of words in document m that are assigned to the topic k
    n_m_k = np.zeros((len(docs), number_topic), dtype=int)

    # длина документа
    # document length
    n_m = np.zeros(len(docs), dtype=int)

    # количество совпадений термина t, присвоенного теме k в коллекции
    # number of occurences of term t assigned to the topic k in the collection
    n_k_t = np.zeros((number_topic, len(dict)), dtype=int)

    # number of terms assigned to the topic k
    # количество терминов присвоенных теме k
    n_k = np.zeros(number_topic, dtype=int)
    # ----------------------------------------
    print('Время в generative process:', time() - time_begin)
    aSample_ = []
    for i, doc in enumerate(docs):
        split_doc = doc.split()
        for j, word in enumerate(split_doc):
            index_word = dict.index(word)
            tf = matrix_tf[i, index_word]
            for cur_word in range(0, tf):
                w = ItemWord()
                w.docID = i
                w.wordID = index_word
                w.topicID = random.randint(0, number_topic - 1)
                aSample_.append(w)

                m = w.docID
                t = w.wordID
                k = w.topicID

                n_m_k[m][k] += 1
                n_m[m] += 1
                n_k_t[k][t] += 1
                n_k[k] += 1

    return aSample_, n_m_k, n_m, n_k_t, n_k


def Gibbs_sampling(aSample, n_m_k, n_m, n_k_t, n_k, dict, ne, type_ne):
    L = 500  # number of iterations

    for iter_L in tqdm(range(0, L)):
        for wIndex, w in enumerate(aSample):
            m = w.docID
            t = w.wordID
            k_old = w.topicID

            # делать один раз в конце
            '''
            topics:
            0 - Адрес
            1 - Дата
            2 - Местоположение
            3 - Деньги
            4 - Имя
            5 - Организация
            6 - Персона
            7 - no_name_1
            8 - no_name_2
            9 - no_name_3
            '''
            # ['2002 гг', '28.06.2006', '1917 г.', '1937 гг', '1920 году', '1917 г.', '14 июля 1937 г.', '1999 г.',
            #  '1925 гг', '1925 гг', '1925 гг', '1925 гг', '1917 г.', '1937 гг', '1927 годов', '1933 гг', '1995 г.',
            #  '1998 г.', 'Г. Российские архетипы', 'с одной стороны', 'с врагами', 'с буржуазным реставраторством',
            #  'с идеологией', 'Г. Обич', 'с помощью', 'С.А', 'С.Б', 'с тем в', 'С. Бочарова', 'с кризисом', 'Г. Дугин',
            #  'г. О', 'Д. Натсак', 'Г. Сироткин', 'С.Агур', 'с превращением', 'идеологическую область', 'С нашей точки',
            #  'с тем и', 'с умственным движением', 'с контрреволюцией', 'с буржуазной идеологией', 'Г. Из', 'С.А', 'С.Б',
            #  'Г.М', 'Д. Шаховского', 'С. Движения', 'Г. Метафизика', 'С. Современная историография', 'С.В', 'Д. Сменовеховство',
            #  'Шугуров М. В.', 'Горшков М.', 'Касьянова К.', 'Вилков А.А.', 'Динь', 'МеОвеОсв И.П.', 'Устрялов', 'Н.В.УСТРЯЛОВА',
            #  'Н.В. Устрялова', 'Николай Васильевич Устрялов', 'Э.Б. Генкина', 'А.И. Линяев', 'Мамай', 'Л.Ф. Шоричев', 'Л.Я. Трифонов',
            #  'О.И. Хохунова', 'Н.А. Королева', 'С.А. Федюкин', 'Э.И. Черняк', 'К.П. Байлов', 'А.В. Квакин', 'А.В. Соловьев',
            #  'Е.О. Обичкина', 'С.Б. Чернышев', 'Н.А. Омельченко', 'Н.В. Устрялова', 'М. Кулагина', 'З.С. Бочарова', 'Н.В. Устрялова',
            #  'Н.В. Устрялова', 'устрялов', 'Сталина', 'А.В. Байлов', 'А.Г. Дугин', 'В.К. Романовский', 'Н.В. Устрялова', 'О. Воробьев',
            #  'Н.В. Устрялова', 'Устрялова', 'А. Иванников', 'В.А. Митрохин', 'Н.В. Работяжев', 'В.Г. Сироткин', 'Н.В. Устрялова',
            #  'М.С.Агур', 'Н.В. Устрялова', 'Н.В. Устрялов', 'Н.В. Устрялов', 'Н.В. Устрялова', 'Устрялова', 'Бубнов А.', 'Устрялове',
            #  'Генкино Э.Б.', 'Линяев А.И.', 'Шоричев Л.Ф.', 'Трифонов Л.Я.', 'ив', 'ФеОюкин С.А.', 'Черняк', 'Батов К.П.', 'Бачурина',
            #  'Попов А.В.', 'Квакин А.В.', 'В.И. Ленин', 'Соловьев А.В.', 'Обичкина Е.О.', 'Кондратьева Т.', 'Чернышев С.Б.',
            #  'Омельченко Н.А.', 'Кулагина Г.М.', 'Бочарова', 'Б. Челышева', 'Д. Шаховского', 'Бочарова З.С.', 'Байлов А.В.',
            #  'Дугин А.Г.', 'Н.В. Устрялов', 'Бочарова З.С.', 'Устрялова', 'Быстрянцева Л. А.', 'Н.В. Устрялова', 'Сталина',
            #  'Н.В. Устрялова', 'Н.Л. Цурикова', 'Н.В. Устрялова', 'С.В. Дмитриевского', 'Иванников П.А.', 'Митрохин В.А.',
            #  'Сироткин В.', 'Троцкий', 'Агурскии М.', 'Вопр', 'Российское общество',
            #  "Российский общественно-политический центр». М.. 1996. Л1>2. С.49. Динь Ты Аоа. Политические конфликты в процессе демократизации общества: Дис. . канл. социол. паук. М.. 2000. С.36.\t'",
            #  'ес', 'Булычёва Саратовский государственный университет', 'современном обществе', 'КПСС в', 'КПСС', 'вовеховствс',
            #  'проф', 'Саратовский государственный университет', 'религиозных организаций', 'ведущих религиозных организации',
            #  'КП', 'данным международной Евангельской миссионерской организации', 'Шугуров М. В.', 'Горшков М.', 'Касьянова К.',
            #  'Вилков А.А.', 'Динь', 'МеОвеОсв И.П.', 'Устрялов', 'Н.В.УСТРЯЛОВА', 'Н.В. Устрялова',
            #  'лидеров этой части российских эмигрантов Николай Васильевич Устрялов', 'Э.Б. Генкина', 'А.И. Линяев', 'Мамай',
            #  'Л.Ф. Шоричев', 'Л.Я. Трифонов', 'О.И. Хохунова', 'Н.А. Королева', 'С.А. Федюкин', 'Э.И. Черняк', 'К.П. Байлов',
            #  'А.В. Квакин', 'А.В. Соловьев', 'Е.О. Обичкина', 'С.Б. Чернышев', 'Н.А. Омельченко', 'Н.В. Устрялова', 'М. Кулагина',
            #  'З.С. Бочарова', 'Н.В. Устрялова', 'Н.В. Устрялова', 'устрялов', 'Сталина', 'А.В. Байлов', 'А.Г. Дугин',
            #  'В.К. Романовский', 'Н.В. Устрялова', 'О. Воробьев', 'Н.В. Устрялова', 'Устрялова', 'А. Иванников', 'В.А. Митрохин',
            #  'Н.В. Работяжев', 'В.Г. Сироткин', 'Н.В. Устрялова', 'М.С.Агур', 'Н.В. Устрялова', 'Н.В. Устрялов', 'Н.В. Устрялов',
            #  'Н.В. Устрялова', 'Устрялова', 'Бубнов А.', 'Устрялове', 'Генкино Э.Б.', 'Линяев А.И.', 'Шоричев Л.Ф.',
            #  'Трифонов Л.Я.', 'ив', 'ФеОюкин С.А.', 'Черняк', 'Батов К.П.', 'Бачурина', 'Попов А.В.', 'Квакин А.В.', 'В.И. Ленин',
            #  'Соловьев А.В.', 'Обичкина Е.О.', 'Кондратьева Т.', 'Чернышев С.Б.', 'Омельченко Н.А.', 'Кулагина Г.М.', 'Бочарова',
            #  'Б. Челышева', 'Д. Шаховского', 'Бочарова З.С.', 'Байлов А.В.', 'Дугин А.Г.', 'Н.В. Устрялов', 'Бочарова З.С.',
            #  'Устрялова', 'Быстрянцева Л. А.', 'Н.В. Устрялова', 'Сталина', 'Н.В. Устрялова', 'Н.Л. Цурикова', 'Н.В. Устрялова',
            #  'С.В. Дмитриевского', 'Иванников П.А.', 'Митрохин В.А.', 'Сироткин В.', 'Троцкий', 'Агурскии М.']

            # обновление статистики (мб нужно делать удаление, если == 0)
            # поправки
            # если тема k_old в док-те встретилась 0 раз, то = 0, иначе -= 1
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
            '''#оригинал_начало
            n_m_k[m][k_old] -= 1
            n_k_t[k_old][t] -= 1
            n_k[k_old] -= 1
            n_m[m] -= 1
            #оригинал_конец'''
            # выбор новой темы
            aP = []
            total = 0.0

            # вычисление по формуле
            for k in range(0, number_topic):
                # если тема k встретилась в коллекции не 0 раз, то...
                if n_k[k] != 0:
                    sum_n_k_T = n_k[k] + beta * len(dict)  # по формуле (79)
                else:
                    sum_n_k_T = beta * len(dict)
                if n_m[m] != 0:
                    sum_n_m_K = n_m[m] + alpha * number_topic  #
                else:
                    sum_n_m_K = alpha * number_topic
                # сколько раз термин t был упомянут в теме k
                n_k_t_K_T = 0
                if n_k_t[k][t] != 0:
                    n_k_t_K_T = n_k_t[k][t]
                # сколько слов в док-те m было присвоено теме k
                n_m_k_M_K = 0
                if n_m_k[m][k] != 0:
                    n_m_k_M_K = n_m_k[m][k]

                # p(k|t,(!k)) = вероятность того, что тема k, описывается термином t отнесенного ко всем темам, кроме k
                p = (n_k_t_K_T + beta) / sum_n_k_T * (n_m_k_M_K + alpha) / sum_n_m_K

                t_ = ItemTopic()
                t_.topicID = k
                t_.probability = p

                aP.append(t_)

                # сумма условной вероятности: вероятность, что тема k описывается термином t
                total += p

            # обновить тему: как только сумма вероятность темы/сумму_условных_вероятностей будет больше
            # наперед заданого числа, считать что термин t относится к теме k_new
            '''
            k_new = -1
            k_last = -1
            r = random.random()
            sum_ = 0.0
            for t_ in aP:
                sum_ += t_.probability / total
                k_last = t_.topicID
                if (sum_ > r):
                    k_new = k_last
                    break
            # если тема не обновилась, то взять последнюю отобранную тему
            if k_new == -1:
                k_new = k_last'''
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

            # проверка на совпадение тем и NE
            # if flag == 0:
            #    flag, check_aSample = check_topic(dict[t], ne[m], k_new, check_aSample, wIndex, type_ne[m])

            aSample[wIndex].topicID = k_new

        for w_iter, w in enumerate(aSample):
            cur.execute('''INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id, check_bool) \
                              VALUES (%s, %s, %s, %s, %s)''', (iter_L, w.docID, w.wordID, w.topicID,
                                                               check_aSample[w_iter].check))
        con.commit()

    for w_iter, w in enumerate(aSample):
        cur.execute('''INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id, check_bool) \
                                      VALUES (%s, %s, %s, %s, %s)''', (L, w.docID, w.wordID, w.topicID,
                                                                       check_aSample[w_iter].check))
    con.commit()
    return

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


# NEW
def normal_word(doc):
    tokenizer = RegexpTokenizer(r'\w+')
    ru_stop = get_stop_words('russian')
    morph = pymorphy2.MorphAnalyzer()
    # clean and tokenize document string
    raw_lower = doc.lower()
    tokens = tokenizer.tokenize(raw_lower)
    # remove stop words from tokens
    stopped_tokens = np.array([k for k in tokens if k not in ru_stop])
    # stem tokens (нормализация слов)
    stemmed_tokens = np.array([morph.normal_forms(l)[0] for l in stopped_tokens])
    # add tokens to list
    text = '|'.join(stemmed_tokens)
    return text


# NEW
def tf_matrix_and_dict_new(cur):
    t_tf = time()
    cur.execute('''SELECT count(*) FROM public.news_rbc''')
    len_db = cur.fetchone()[0]
    matrix_tf_2 = np.array([])
    dict_words = set()
    for i in tqdm(range(0,6000)):
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
        # ОШИБКА в сплит нужно '|'
        len_text_without_ne = len(text.split('|'))
        num_ne = np.array([len_text_without_ne + x for x in range(len(ne))])
        # добавить в нормализованный текст NE
        text = text + '|' + '|'.join(ne)
        # -=-получить словарь-=-
        # разбить документ на слова по разделителю '|'
        split_doc = text.split('|')
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
        cur.execute('INSERT INTO public.news_rbc_in_normal_form'
                    '(id, norm_text, ne, span_ne, num_ne_in_text, type_ne)'
                    'VALUES (%s, %s, %s, %s, %s, %s)',
                    (i, text, ne, span_ne, num_ne, type_ne))
        con.commit()
    ordered_dict_new = np.array([x for x in dict_words])
    print('Всего документов: {}'.format(len(matrix_tf_2)))
    print('Длина словаря = {}'.format(len(ordered_dict_new)))
    for i,word in enumerate(ordered_dict_new):
        cur.execute('INSERT INTO public.dictionary_news_rbc (id, word) VALUES (%s, %s)', (i,str(word)))
    con.commit()
    for i in tqdm(range(len(matrix_tf_2))):
        cur.execute('SELECT id FROM public.news_rbc_in_normal_form ORDER BY id ASC LIMIT 1 OFFSET %s',
                    (i,))
        doc_id = cur.fetchone()[0]
        for word_id in range(len(ordered_dict_new)):
            word = str(ordered_dict_new[word_id])
            tf = int(matrix_tf_2[i].get(word, 0))
            cur.execute('INSERT INTO public.matrix_tf (doc_id, word_id, tf, word) VALUES (%s, %s,%s, %s)',
                        (doc_id, word_id, tf, word))
    con.commit()
    print('Время создания matrix_tf и ordered_dict: {}'.format(time() - t_tf))
    return matrix_tf_2, ordered_dict_new


# NEW
def generative_process_new(dict_full, cur):
    t_generative = time()
    # получить кол-во элементов в БД = всего док-тов
    cur.execute('''SELECT count(*) FROM public.news_rbc_in_normal_form''')
    len_docs = cur.fetchone()[0]
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
    aSample_ = np.array([])
    for i in tqdm(range(len_docs)):
        cur.execute('SELECT id, norm_text, ne, span_ne, num_ne_in_text, type_ne '
                    'FROM public.news_rbc_in_normal_form ORDER BY id ASC'
                    'LIMIT 1 OFFSET %s', (i,))
        row = cur.fetchone()
        doc_id = row[0]
        doc = row[1]
        #ne = row[2]
        #span_ne = row[3]
        num_ne = row[4]
        type_ne = row[5]
        split_doc = doc.split('|')

        for j, word in enumerate(split_doc):
            '''if word in matrix_tf[i]:
                tf = matrix_tf[i][word]
            else:
                tf = 0'''
            word_id = np.where(dict_full == word)[0][0]
            '''for re_word in range(0, tf):'''
            w = ItemWord()
            w.docID = doc_id
            w.wordID = word_id
            w.topicID = random.randint(0, number_topic - 1)
            if j in num_ne:
               w._ne.is_ne = True
               w._ne._type_ne = what_is_number_type_ne(type_ne[num_ne.index(j)])
            else:
               w._ne.is_ne = False
               w._ne._type_ne = None
            aSample_ = np.append(aSample_, w)

            m = w.docID
            t = w.wordID
            k = w.topicID

            n_m_k[m][k] += 1
            n_m[m] += 1
            n_k_t[k][t] += 1
            n_k[k] += 1
        #del row, doc_id, ...

    print('Время в generative process: {}'.format(time() - t_generative))
    return aSample_, n_m_k, n_m, n_k_t, n_k


# NEW
def Gibbs_sampling_new(aSample, n_m_k, n_m, n_k_t, n_k, dict_full):
    time_gibbs = time()
    L = 500  # number of iterations
    for iter_L in tqdm(range(0, L)):
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
            '''#оригинал_начало
            n_m_k[m][k_old] -= 1
            n_k_t[k_old][t] -= 1
            n_k[k_old] -= 1
            n_m[m] -= 1
            #оригинал_конец'''
            # выбор новой темы
            aP = np.array([])
            total = 0.0

            # вычисление по формуле
            # ВОТ ЭТО НУЖНО ПОПРАВИТЬ И СДЕЛАТЬ ЧУТЬ ЛИ НЕ В ОДНУ СТРОЧКУ
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
                if n_k[k] != 0:
                    sum_n_k_T = n_k[k] + beta * len(dict_full)  # по формуле (79)
                else:
                    sum_n_k_T = beta * len(dict_full)
                if n_m[m] != 0:
                    sum_n_m_K = n_m[m] + alpha * number_topic  #
                else:
                    sum_n_m_K = alpha * number_topic
                # сколько раз термин t был упомянут в теме k
                n_k_t_K_T = 0
                if n_k_t[k][t] != 0:
                    n_k_t_K_T = n_k_t[k][t]
                # сколько слов в док-те m было присвоено теме k
                n_m_k_M_K = 0
                if n_m_k[m][k] != 0:
                    n_m_k_M_K = n_m_k[m][k]

                # p(k|t,(!k)) = вероятность того, что тема k, описывается термином t
                p = (n_k_t_K_T + beta) / sum_n_k_T * (n_m_k_M_K + alpha) / sum_n_m_K

                t_ = ItemTopic()
                t_.topicID = k
                t_.probability = p

                aP = np.append(aP,t_)

                # сумма условной вероятности: вероятность, что тема k описывается термином t
                total += p
            '''
            # обновить тему: как только сумма вероятность темы/сумму_условных_вероятностей будет больше
            # наперед заданого числа, считать что термин t относится к теме k_new
            # версия Добрынина В.Ю.

            k_new = -1
            k_last = -1
            r = random.random()
            sum_ = 0.0
            for t_ in aP:
                sum_ += t_.probability / total
                k_last = t_.topicID
                if (sum_ > r):
                    k_new = k_last
                    break
            # если тема не обновилась, то взять последнюю отобранную тему
            if k_new == -1:
                k_new = k_last'''
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

        for w in aSample:
            cur.execute('''INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id) \
                              VALUES (%s, %s, %s, %s, %s)''', (iter_L, w.docID, w.wordID, w.topicID))
        con.commit()

    # записать последнюю итерацию
    for w in aSample:
        cur.execute('INSERT INTO public."LDA_samples" (sample_id, doc_id, word_id, topic_id)'
                    ' VALUES (%s, %s, %s, %s, %s)', (L, w.docID, w.wordID, w.topicID))
    con.commit()

    print('Время в Gibbs_Sampling: {}'.format(time() - time_gibbs))
    return



if __name__ == '__main__':
    time_begin = time()
    # подключаемся к базе
    con = psycopg2.connect(database="texts_politics", user="postgres", password="197346qaz", host="127.0.0.1",
                           port="5432", )
    # создаем курсор для работы с базой данных
    cur = con.cursor()

    number_topic = 100
    alpha = 50 / number_topic
    beta = 0.01

    tf_mat, ordered_dict = tf_matrix_and_dict_new(cur)
    aSample, n_m_k, n_m, n_k_t, n_k = generative_process_new(ordered_dict,cur)
    Gibbs_sampling_new(aSample, n_m_k, n_m, n_k_t, n_k, ordered_dict)
    print('Время исполнения программы:', time() - time_begin)
    exit()


    data_text, data_ne, type_ne = read_files(cur)
    docs_in_normal = normal_form_word(data_text)
    dictionary, number_words_of_docs = get_dict(docs_in_normal)
    ordered_dict, indexs = indexing_words_of_dict(dictionary)
    matrix_tf_word_in_docs = tf_matrix(docs_in_normal, ordered_dict)

    number_topic = 10
    alpha = 50 / number_topic
    beta = 0.01
    aSample, n_m_k, n_m, n_k_t, n_k = generative_process(matrix_tf_word_in_docs, docs_in_normal, ordered_dict)
    '''
    topics:
    0 - Адрес
    1 - Дата
    2 - Местоположение
    3 - Деньги
    4 - Имя
    5 - Организация
    6 - Персона
    7 - no_name_1
    8 - no_name_2
    9 - no_name_3
    '''
    print('word = {}'.format(ordered_dict[19706]))
    print('word = {}'.format(ordered_dict[8299]))
    print('word = {}'.format(ordered_dict[1722]))
    print('word = {}'.format(ordered_dict[17793]))
    print('word = {}'.format(ordered_dict[413]))

    '''for i in range(len(aSample)):
        print('i = {}, docID = {}, wordID = {}, topicID = {}, word = {}'.format(i,
                                                                                aSample[i].docID,
                                                                                aSample[i].wordID,
                                                                                aSample[i].topicID,
                                                                                ordered_dict[aSample[i].wordID]))'''

    Gibbs_sampling(aSample, n_m_k, n_m, n_k_t, n_k, ordered_dict, data_ne, type_ne)
    # 20 итераций - 9 часов 13 минут
    print('Время исполнения программы:', time() - time_begin)
