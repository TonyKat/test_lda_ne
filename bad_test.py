import numpy as np
from collections import Counter

ALPHA = 100
BETA = 5
ITERATIONS = 1000

def read_corpuses(*filenames):
    words = Counter()
    for corpus_file in filenames:
        with open(corpus_file) as corpus:
            words.update(word for line in corpus for word in line.split())
    return words

def compute_image(vocabulary, corpus_filename):
    with open(corpus_filename) as corpus:
        return [vocabulary.index(word) for line in corpus for word in line.split()]

def init_LDA(images, M, V):
    I = len(images)
    Na = np.zeros((I, M)) + ALPHA     # number of words for each document, topic combo i.e 11, 12,13 -> 21,22,23 array.
    # количество слов в документе отнесенных теме M
    Nb = np.zeros(I) + M*ALPHA        # number of words in each image
    # количество слов к каждом документе
    Nc = np.zeros((M, V)) + BETA      # word count of each topic and vocabulary, times the word is in topic M and is of vocab number 1,2,3, etc..
    # матрица: сколько раз слово было отнесено теме M
    Nd = np.zeros(M) + V*BETA         # number of words in each topic
    # количество слов в каждой теме
    def inner(i, w):
        m = np.random.randint(0, M)
        Na[i, m] += 1
        Nb[i] += 1
        Nc[m, w-1] += 1
        Nd[m] += 1
        return m

    return Na, Nb, Nc, Nd, [[inner(i, w) for w in image] for i, image in enumerate(images)]

def LDA(topics, *filenames):
    words = read_corpuses(*filenames)
    vocabulary = words.keys()

    images = [compute_image(vocabulary, corpus) for corpus in filenames]

    Na, Nb, Nc, Nd, topic_of_words_per_image = init_LDA(images, topics, len(vocabulary))

    #Gibbs Sampling
    probabilities = np.zeros(topics)
    for _ in range(ITERATIONS):

        for i, image in enumerate(images):
            topic_per_word = topic_of_words_per_image[i]
            for n, w in enumerate(image):
                m = topic_per_word[n]

                Na[i, m] -= 1
                Nb[i] -= 1
                Nc[m, w-1] -= 1
                Nd[m] -= 1

                # computing topic probability
                probabilities[m] = Na[i, m] * Nc[m, w-1]/(Nb[i] * Nd[m])
                # choosing new topic based on this
                q = np.random.multinomial(1, probabilities/probabilities.sum()).argmax()
                # assigning word to topic
                topic_per_word[n] = q

                Na[i, q] += 1
                Nb[i] += 1
                Nc[q, w-1] += 1
                Nd[q] += 1

    distances = Nc/Nd[:, np.newaxis] #Words by Topic and printing
    return distances, vocabulary, words

if __name__ == '__main__':
    topics = 2
    #Add as many filenames as needed, like LDA(topics, 'corpus1.txt', 'corpus2.txt', 'corpus3.txt')
    distances, vocabulary, words_count = LDA(topics, 'corpus.txt')

    for topic in range(topics):
        for word_index in np.argsort(-distances[topic])[:20]:
            word = vocabulary[word_index]
            print("Topic", topic, word, distances[topic, word_index], words_count[word])