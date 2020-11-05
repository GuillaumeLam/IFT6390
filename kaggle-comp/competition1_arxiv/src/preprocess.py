import csv
import re
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import numpy as np

from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# list of word types (nouns and adjectives) to leave in the text
defTags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']#, 'RB', 'RBS', 'RBR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# functions to determine the type of a word
def is_noun(tag):
	return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
	return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
	return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
	return tag in ['JJ', 'JJR', 'JJS']

# transform tag forms
def penn_to_wn(tag):
	if is_adjective(tag):
		return nltk.stem.wordnet.wordnet.ADJ
	elif is_noun(tag):
		return nltk.stem.wordnet.wordnet.NOUN
	elif is_adverb(tag):
		return nltk.stem.wordnet.wordnet.ADV
	elif is_verb(tag):
		return nltk.stem.wordnet.wordnet.VERB
	return nltk.stem.wordnet.wordnet.NOUN


def clean(comment_string, lemmatizer):
        clean_tokens = []

        token = comment_string
        
        for thing, tag in nltk.pos_tag(token.split()):
            if (thing not in string.punctuation):
                clean_tokens.append(lemmatizer.lemmatize(thing, penn_to_wn(tag)))
        
        clean_tokens = [word for word in clean_tokens if word not in stopwords.words('english')]
            
        token = ' '.join(clean_tokens)
                
        matches = re.findall(r'\\\w*', token)
        matches = [re.sub(r'\\', '', word) for word in list(set(matches))]
        token = re.sub(r'\\\w*', ' ', token)
        token = token + ' '.join(matches)
        
        token = re.sub(r'\[', ' ', token)
        token = re.sub(r'\]', ' ', token)

        
        token = re.sub(r'\n', ' ', token)

        token = re.sub(r'\?', ' ', token)
        token = re.sub(r'\"', ' ', token)
        token = re.sub(r'\!', ' ', token)
        token = re.sub(r'\,', ' ', token)
        token = re.sub(r'\.', ' ', token)
        token = re.sub(r'\:', ' ', token)
        token = re.sub(r'\;', ' ', token)
        token = re.sub(r'\)', ' ', token)
        token = re.sub(r'\(', ' ', token)

        token = re.sub(r"\'", ' ', token)
        token = re.sub(r'\+', ' ', token)
        token = re.sub(r"\-", ' ', token)
        token = re.sub(r"\~", ' ', token)
        token = re.sub(r"\*", ' ', token)
        token = re.sub(r"\&", ' ', token)
        token = re.sub(r"\{", ' ', token)
        token = re.sub(r"\}", ' ', token)
        token = re.sub(r"\|", ' ', token)
        token = re.sub(r"\/", ' ', token)
        token = re.sub(r"\#", ' # ', token)
        token = re.sub(' +', ' ', token)

        token = re.sub(r' 200\d ', ' [year]', token)
        token = re.sub(r' 20\d\d ', ' [year]', token)
        token = re.sub(r' 199\d ', ' [year]', token)

        token = re.sub(r' \d+', ' [number] ', token)

        token = token.lower()
        
#         token = re.sub(r'\$( )*\$', ' ', token)
# 
#         token = re.sub(r'\$.*\$', ' ', token)
# 

        token = re.sub(r'\_', ' ', token)

        return ' '.join(token.split())


def pre_process(file_path, data = 'train', vectorizer = 'tfidf', max_features = None, existing_vectorizer=None):    
    ARXIV = ['astro-ph', 'astro-ph.CO', 'astro-ph.GA', 'astro-ph.SR',
       'cond-mat.mes-hall', 'cond-mat.mtrl-sci', 'cs.LG', 'gr-qc',
       'hep-ph', 'hep-th', 'math.AP', 'math.CO', 'physics.optics',
       'quant-ph', 'stat.ML']
    
    lemmatizer = WordNetLemmatizer()
    
    if existing_vectorizer:
        vectorizer = existing_vectorizer
    else:
        if vectorizer == 'tfidf':
            vectorizer = TfidfVectorizer(max_features = max_features)
        elif vectorizer == 'count':
            vectorizer = CountVectorizer(max_features = max_features)
        elif vectorizer == 'binary':
            vectorizer = CountVectorizer(max_features = max_features, binary = True)


    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        colnames = next(csv_reader)

        print('cleaning...')
        if data == 'train':
            raw_data = [[_, clean(comment, lemmatizer), ARXIV.index(cl)] for _, comment, cl in list(csv_reader)]
            X, y = np.array(raw_data)[:, 1], np.array(raw_data)[:, 2]
            
        elif data == 'test':
            raw_data = [[_, clean(comment, lemmatizer)] for _, comment in list(csv_reader)]
            X, y = np.array(raw_data)[:, 1], None

    if data == 'train':

        print('vectorizing...')
        X = vectorizer.fit_transform(X).toarray()
        
        print('done!')
        return X, y, vectorizer
      
    elif data == 'test':
        print('vectorizing...')
        X = vectorizer.transform(X).toarray()

        print('done!')
        return X, None
