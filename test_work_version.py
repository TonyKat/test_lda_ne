# -*- coding: utf-8 -*-
import psycopg2
import re
import numpy as np

from time import time
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from functools import reduce

from natasha import (AddressExtractor, DatesExtractor, LocationExtractor, MoneyExtractor,
    NamesExtractor, OrganisationExtractor, PersonExtractor)

from natasha.grammars.address import Address
from natasha.grammars.date import Date
from natasha.grammars.location import Location
from natasha.grammars.money import Money
from natasha.grammars.name import Name
from natasha.grammars.organisation import Organisation
from natasha.grammars.person import Person


class Element_of_entity:
    def __init__(self, text = None, named_entities = None):
        self.text = text
        self.named_entities = named_entities
        #self.span_before_ne = span_before_ne


def split_text_into_sentences(texts):
    split_text = []
    for text in texts:
        sentences = []
        for text in re.split(r'(?<=[.!?…]) ', text):
            sentences.append(text)
        split_text.append(sentences)
    return split_text


def send_to_db():
    conn = psycopg2.connect(database="postgres", user="postgres", password="197346qaz", host="127.0.0.1", port="5432", )
    # создаем курсор для работы с базой данных
    curr = conn.cursor()
    curr.execute(
        'SELECT full_text FROM public.papers WHERE (full_text != (%s) and is_downloads = True and source = 5)'
        ' ORDER BY full_text ASC LIMIT 20000 OFFSET(%s)', ('', 0))
    rows = curr.fetchall()
    conn.close()

    print('len(rows) =',len(rows))

    con = psycopg2.connect(database="texts_politics", user="postgres", password="197346qaz", host="127.0.0.1", port="5432", )
    cur = con.cursor()

    for i in range(len(rows)):
        cur.execute("INSERT INTO public.news_of_politics (id,text,named_entities,two_words_before_ne,span_ne,span_before_ne) \
              VALUES (%s, %s, %s, %s, %s, %s)", (i, rows[i], ['ne_' + str(i)],
                                                 ['two_before_ne_' + str(i)], [i, i + 1], [i, i + 1]))
        #cur.execute("INSERT INTO public.news_of_politics (named_entities,two_words_before_ne,span_ne,span_before_ne) \
        #              VALUES (%s, %s, %s, %s)", ('cry','try',[0],[10]))#['sky', 'cry'], ['one_word', 'two_word'],[0,10],[20,30]))
    print('Время =', time() - time_begin)
    con.commit()
    con.close()
    return True


def get_data_from_db_politics(cur):
    cur.execute("SELECT text FROM public.news_of_politics ORDER BY id ASC LIMIT 100")
    rows = cur.fetchall()
    return rows


def what_is_ne(ne):
    if type(ne) == Address:
        return 'Адрес'
    if type(ne) == Date:
        return 'Дата'
    if type(ne) == Location:
        return 'Местоположение'
    if type(ne) == Money:
        return 'Деньги'
    if type(ne) == Name:
        return 'Имя'
    if type(ne) == Organisation:
        return 'Организация'
    if type(ne) == Person:
        return 'Персона'


def split_without_punctuation(text_for_split):
    punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':',
                   ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
                   '—', '«', '»', '…', '–']

    wordList = text_for_split.split()
    number_word_punctuation = []
    number_word = 0
    for word in wordList:
        try:
            if word in punctuation:
                number_word_punctuation.append(number_word)
                number_word += 1
                continue
            punct_begin = 0
            while punct_begin < len(word) and word[punct_begin] in punctuation:
                punct_begin += 1
            punct_end = 0
            while punct_end < len(word) and word[len(word) - 1 - punct_end] in punctuation:
                punct_end += 1
            if word[punct_begin:len(word) - punct_end] == '':
                number_word_punctuation.append(number_word)
                number_word += 1
                continue
            wordList[number_word] = word[punct_begin:len(word) - punct_end]
            number_word += 1
        except:
            print('number_word =', number_word)
            print('word =', word)

    # удаление слов состоящих из зн.преп
    #for i in range(len(number_word_punctuation)):
    #    wordList.pop(number_word_punctuation[i] - i)
    #for x in number_word_punctuation:
    #    wordList[x] = None

    return wordList


def number_punctuation(text):
    punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':',
                   ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
                   '—', '«', '»', '…', '–']
    number_punct_begin = 0
    number_punct_end = 0

    for char in text:
        if char in punctuation:
            number_punct_begin += 1
            continue
        else:
            for char_end in text[len(text):number_punct_begin:-1]:
                if char_end in punctuation:
                    number_punct_end += 1
            break

    return [number_punct_begin, number_punct_end]


def check_words(words, text_1, text_2):
    list_text_1 = text_1.split()
    list_text_2 = text_2.split()
    empty_array = []
    print(words)
    words = ['', words[0], words[1], words[2]]
    print(words)
    for i in range(len(words)):
        if words[i] == '' and i == 0:
            j = 1
            new_words = split_without_punctuation(' '.join([list_text_1[len(list_text_1) - (j + 2)]]))
            while new_words[0] == '':

                j += 1
                new_words = split_without_punctuation(' '.join([list_text_1[len(list_text_1) - (j + 2)]]))

        if words[i] == '' and i == 1:
            j = 1
            new_words = split_without_punctuation(' '.join([list_text_1[len(list_text_1) - (j + 3)]]))
            while new_words[0] == '':
                j += 1
                new_words = split_without_punctuation(' '.join([list_text_1[len(list_text_1) - (j + 3)]]))


    return


def two_words_before_and_after(text, span_ne):
    '''
        если 2 слова до и после NE состоят из символов,
         то они не запоминаются
    '''
    # разбить текст на 2 части без NE
    text_1 = text[0:span_ne[0]]
    text_2 = text[span_ne[1]:len(text)]
    # получить 2 слова до и после NE
    words_1_2 = text_1.split()[len(text_1.split()) - 2:len(text_1.split())]
    words_2_2 = text_2.split()[0:2]
    # проверка на пустоту
    '''начало'''
    if words_1_2 == []:
        words_1_2 = ['', '']
    if len(words_1_2) == 1:
        words_1_2 = ['', words_1_2[0]]
    if words_2_2 == []:
        words_2_2 = ['', '']
    if len(words_2_2) == 1:
        words_2_2 = [words_2_2[0], '']
    '''конец'''
    words_before_after = [words_1_2[0], words_1_2[1], words_2_2[0], words_2_2[1]]
    # получить 2 слова до и после NE без пунктуации
    words_without_punct = split_without_punctuation(' '.join([words_before_after[0], words_before_after[1],
                                                              words_before_after[2], words_before_after[3]]))
    for i in range(4):
        if words_before_after[i] == '':
            words_without_punct.insert(i, '')
    # узнать кол-во зн.преп. в слове
    puncts = []
    for k in range(len(words_before_after)):
        puncts.append(number_punctuation(words_before_after[k]))
    '''
        1) если None - запоминать кол-во зн.преп. из которого состояло слово
        2) считать пробелы и запоминать их позиции, чтобы найти правильный span слов

        или
        попробовать искать индекс слова
        сделать условие на пунктуацию 
    '''


    relative_len_ = len(words_before_after[0]) + len(words_before_after[1]) # len слов с пунктуацией
    index_1 = text.find(words_without_punct[0], span_ne[0] - relative_len_ - 2)
    index_2 = text.find(words_without_punct[1], index_1 + len(words_without_punct[0]))
    index_3 = text.find(words_without_punct[2], span_ne[1])
    index_4 = text.find(words_without_punct[3], span_ne[1] + len(words_without_punct[2]))
    span_words_without_punct = [[index_1, index_1 + len(words_without_punct[0])],
                      [index_2, index_2 + len(words_without_punct[1])],
                      [index_3, index_3 + len(words_without_punct[2])],
                      [index_4, index_4 + len(words_without_punct[3])]]

    # получить интервалы этих слов без зн.преп.
    relative_len_1 = len(words_before_after[0]) + len(words_before_after[1])
    '''
    span_words_without_punct = [
        [span_ne[0] - relative_len_1 - 2 + puncts[0][0],
         span_ne[0] - relative_len_1 - 2 + len(words_before_after[0]) - puncts[0][1]],
        [span_ne[0] - len(words_before_after[1]) - 1 + puncts[1][0],
         span_ne[0] - len(words_before_after[1]) - 1 + len(words_before_after[1]) - puncts[1][1]],
        [span_ne[1] + 1 + puncts[2][0],
         span_ne[1] + 1 + len(words_before_after[2]) - puncts[2][1]],
        [span_ne[1] + 1 + len(words_before_after[2]) + 1 + puncts[3][0],
         span_ne[1] + 2 + len(words_before_after[2]) + len(words_before_after[3]) - puncts[3][1]]]
    '''
    # с зн.преп.
    span_words_full = [
        [span_ne[0] - relative_len_1 - 2,
         span_ne[0] - relative_len_1 - 2 + len(words_before_after[0])],
        [span_ne[0] - len(words_before_after[1]) - 1,
         span_ne[0] - len(words_before_after[1]) - 1 + len(words_before_after[1])],
        [span_ne[1] + 1,
         span_ne[1] + 1 + len(words_before_after[2])],
        [span_ne[1] + 1 + len(words_before_after[2]) + 1,
         span_ne[1] + 2 + len(words_before_after[2]) + len(words_before_after[3])]]

    # тип (тема) 2 слов до и после
    #type_words.append(['<Имя>','<Дата>','<Деньги>','<Имя>'])



    # на выходе получаем:
    # массив words - 2 слова до и после
    # массив span_before_ne - расположение этих слов в тексте
    # массив type_words - типы ne этих слов

    return words_without_punct, span_words_without_punct


'''начало'''
def get_ne2(match,text_doc):
    return text_doc[match.span[0]:match.span[1]]

def get_ne(match):
    return text_paper[match.span[0]:match.span[1]]

def get_span_ne(match):
    return [match.span[0], match.span[1]]

def get_type_ne(match):
    return what_is_ne(match.fact)

def get_words(match):
    words_without_punct, span_words_without_punct = two_words_before_and_after(text_paper, match.span)
    return words_without_punct

def get_span_words(match):
    words_without_punct, span_words_without_punct = two_words_before_and_after(text_paper, match.span)
    return span_words_without_punct
'''конец'''


def send_to_db_ne(cur, con):
    pool = ThreadPool(10)
    for i in tqdm(range(100,120)): # 41, 48, 53, [87:99]
        ne_full = []
        span_ne_full = []
        type_ne_full = []
        words_full = []
        span_words_full = []
        cur.execute("SELECT text FROM public.news_of_politics ORDER BY id ASC LIMIT 1 OFFSET %s", (i, ))
        data = cur.fetchone()
        try:
            for extr in extractors:
                global text_paper
                text_paper = data[0]
                matches = extr(text_paper)
                ne = pool.map(get_ne, matches)
                span_ne = pool.map(get_span_ne, matches)
                type_ne = pool.map(get_type_ne, matches)
                words = pool.map(get_words, matches)
                span_words = pool.map(get_span_words, matches)
                '''
                print('len(ne) =', len(ne))
                print(ne)
                print(span_ne)
                print(type_ne)
                print(words)
                print(span_words)
                print(text_paper[3140:3148], text_paper[3150:3154], text_paper[3162:3164], text_paper[3165:3168])
                print(text_paper[8257:8264], text_paper[8265:8267],text_paper [8283:8285], text_paper[8286:8293])
                print(text_paper[20244:20251], text_paper[20252:20254], text_paper[20262:20263], text_paper[20264:20267])

                print('Время =', time() - time_begin)
                '''

                ne_full.append(ne)
                span_ne_full.append(span_ne)
                type_ne_full.append(type_ne)
                words_full.append(words)
                span_words_full.append(span_words)
        except:
            print('i =', i)
        if len(ne_full) != 0:
            ne_for_db = reduce(lambda x, y: x + y, ne_full)
            span_ne_for_db = reduce(lambda x, y: x + y, span_ne_full)
            type_ne_for_db = reduce(lambda x, y: x + y, type_ne_full)
            words_for_db = reduce(lambda x, y: x + y, words_full)
            span_words_for_db = reduce(lambda x, y: x + y, span_words_full)

        if len(ne_for_db) != 0:
            cur.execute('UPDATE public.news_of_politics '
                        'SET named_entities=%s, two_words_before_ne=%s, '
                        'span_ne=%s, type_ne=%s, span_before_ne=%s '
                        'WHERE id=%s;', (ne_for_db, words_for_db, span_ne_for_db,
                                         type_ne_for_db, span_words_for_db, i))
            con.commit()

    print('Сделано!')

    pool.close()
    pool.join()
    return True


def write_to_file(cur):
    for i in range(1000):
        cur.execute("SELECT text FROM public.news_of_politics ORDER BY id ASC LIMIT 1 OFFSET %s", (i,))
        text = cur.fetchone()[0]
        with open('data_politics_txt\\' + 'text_paper_politic_' + str(i) + '.txt', 'w', encoding='utf-8') as output_file:
            try:
                output_file.write(text + '\n')
            except:
                print('Ошибка записи в файл!')
    return


def send_to_db_news_rbc(extractors):
    con = psycopg2.connect(database="texts_politics", user="postgres", password="197346qaz", host="127.0.0.1",
                           port="5432", )
    cur = con.cursor()
    #cur.execute('''SELECT full_text FROM public.news_rbc ORDER BY id ASC''')
    #rows = cur.fetchall() # - 9540
    pool = ThreadPool(10)
    for i in tqdm(range(5839,9540)): #2294
        ne_full = []
        span_ne_full = []
        type_ne_full = []
        cur.execute("SELECT full_text FROM public.news_rbc WHERE id=%s", (i,))
        data = cur.fetchone()
        try:
            for extr in extractors:
                global text_paper
                text_paper = data[0]
                matches = extr(text_paper)
                ne = pool.map(get_ne, matches)
                span_ne = pool.map(get_span_ne, matches)
                type_ne = pool.map(get_type_ne, matches)

                ne_full.append(ne)
                span_ne_full.append(span_ne)
                type_ne_full.append(type_ne)
        except:
            print('i =', i)
            continue
        if len(ne_full) != 0:
            ne_for_db = reduce(lambda x, y: x + y, ne_full)
            span_ne_for_db = reduce(lambda x, y: x + y, span_ne_full)
            type_ne_for_db = reduce(lambda x, y: x + y, type_ne_full)

        if len(ne_for_db) != 0:
            cur.execute('UPDATE public.news_rbc '
                        'SET ne=%s, span_ne=%s, type_ne=%s'
                        'WHERE id=%s;', (ne_for_db, span_ne_for_db,
                                         type_ne_for_db, i))
            con.commit()
    con.commit()
    con.close()
    pool.close()
    pool.join()
    return


def send_db_pool_map(doc):
    extractors = [AddressExtractor(), DatesExtractor(), LocationExtractor(), MoneyExtractor(), NamesExtractor(),
                  OrganisationExtractor(), PersonExtractor()]

    pool_local = ThreadPool(10)
    ne_full = []
    span_ne_full = []
    type_ne_full = []
    try:
        for extr in extractors:
            global text_paper
            text_paper = doc
            matches = extr(text_paper)
            ne = pool_local.starmap(get_ne2, zip(matches,[doc for x in range(len(matches))]))
            span_ne = pool_local.map(get_span_ne, matches)
            type_ne = pool_local.map(get_type_ne, matches)

            ne_full.append(ne)
            span_ne_full.append(span_ne)
            type_ne_full.append(type_ne)
    except:
        print('Ошибка! Примерный номер =', '?')
    pool_local.close()
    pool_local.join()
    if len(ne_full) != 0:
        ne_for_db = reduce(lambda x, y: x + y, ne_full)
        span_ne_for_db = reduce(lambda x, y: x + y, span_ne_full)
        type_ne_for_db = reduce(lambda x, y: x + y, type_ne_full)
        '''if len(ne_for_db) != 0:
            cur.execute('UPDATE public.news_rbc '
                        'SET ne=%s, span_ne=%s, type_ne=%s'
                        'WHERE id=%s;', (ne_for_db, span_ne_for_db,
                                         type_ne_for_db, num))
            con.commit()'''
        return [ne_for_db, span_ne_for_db, type_ne_for_db]
    else:
        return [0, 0, 0]



if __name__ == '__main__':
    time_begin = time()
    # экстракторы
    extractors = [AddressExtractor(), DatesExtractor(), LocationExtractor(), MoneyExtractor(), NamesExtractor(),
                  OrganisationExtractor(), PersonExtractor()]
    send_to_db_news_rbc(extractors) # 292.92 секунды
    # ошибки: 6911;7168;7561;8246;8539;8691;9211
    exit()
    '''con = psycopg2.connect(database="texts_politics", user="postgres", password="197346qaz", host="127.0.0.1",
                           port="5432", )
    cur = con.cursor()
    pool = ThreadPool(10) # было ошибок 8  - 2459? 2779 = []
    for i in tqdm(range(5059,9540,10)): # 296.92 секунды # с3347по3357 не делал
        # обработало 5839 строк, из них 120 строк не обработаных
        cur.execute("SELECT full_text FROM public.news_rbc ORDER BY id ASC LIMIT 10 OFFSET %s", (i,))
        data = cur.fetchall()
        docs = [x[0] for x in data]
        #new_form = pool.starmap(send_db_pool_map, zip(docs,[i_num for i_num in range(i,i+10)]))
        new_form = pool.map(send_db_pool_map,docs) # 281.43 секунды | 293.59
        t_for = time()
        for j,row in enumerate(new_form):
            cur.execute('UPDATE public.news_rbc '
                        'SET ne=%s, span_ne=%s, type_ne=%s'
                        'WHERE id=%s;', (row[0], row[1],
                                         row[2], i+j))
            con.commit()
    pool.close()
    pool.join()
    con.close()'''
    exit()

    #send_to_db()
    # подключаемся к базе
    con = psycopg2.connect(database="texts_politics", user="postgres", password="197346qaz", host="127.0.0.1",
                           port="5432", )
    # создаем курсор для работы с базой данных
    cur = con.cursor()
    #send_to_db_ne(cur, con)
    #data = get_data_from_db_politics(cur)
    write_to_file(cur)

    print('Время =', time() - time_begin)
    exit()
