import nltk
import re
import nltk.corpus
import nltk.tokenize
from nltk.stem import RSLPStemmer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('rslp')


# Separates the words of the input sentence into items of a list.
def tokenize(sentence_string):
    #print(sentence_string) # Debug
    sentence_string = sentence_string.lower()
    preview_tokenized_list = nltk.word_tokenize(sentence_string)
    tokenized_list = []
    for word in preview_tokenized_list:

        if '.' in list(word):# Verifica se existem palavras com pontos grudados
            tokenized_words = word.replace("."," ")  # Substitui ponto por espaço para separar palavras (caso esteja entre duas ou mais palavras)
            tokenized_word = nltk.word_tokenize(tokenized_words) # Generaliza para o caso de mais de uma palavra tokenizando
            if tokenized_word != []:
                for element in tokenized_word:
                    tokenized_list.append(element)
        else:
            tokenized_list.append(word)

    words = [word for word in tokenized_list if word.isalpha()]
    return words

# Reduce the word to its base form.
def stemming(tokenized_list):
    stemmer = RSLPStemmer()
    stemmed_list = []
    for word in tokenized_list:
        stemmed_list.append(stemmer.stem(word.lower()))
    return stemmed_list

# Removes unnecessary words (subjects, connectors, etc).
# For list of stopwords: stopwords = nltk.corpus.stopwords.words('english')
def removeStopWords(stemmed_list):
    stop_words = nltk.corpus.stopwords.words('english')
    final_stop_words = set([word for word in stop_words])
    non_stop_list = []
    for word in stemmed_list:
        if word not in final_stop_words:
            non_stop_list.append(word)
    return non_stop_list


# Calls necessary pre-processing functions to prepare the report text for the nlp model.
def preProcess(report_text):
    processed_text = tokenize(report_text)
    processed_text = removeStopWords(processed_text)
    return processed_text

