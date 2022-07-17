import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stemming(word):
    return stemmer.stem(word.lower())

def bagofwords(tk_sentence, all_words):
    tk = [stemming(w) for w in tk_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for i,word in enumerate(all_words):
        if word in tk:
            bag[i] = 1.0

    return bag

""" this is working
    
    words = ["hello", "hi", "lol", "crazy"]
    sentence = ["hello", "lol", "this", "crazy"]

    bag = bagofwords(sentence, words)
    print(bag)"""