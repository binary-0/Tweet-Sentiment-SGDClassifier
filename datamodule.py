import numpy as np
import pandas as pd
import re
import math
import nltk
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SNSDataset:
    def __init__(self, config=None):
        self.config = config
        self.vocab = None

        # load data
        self.train_data = self._load_data('train')
        self.test_data = self._load_data('test')

        # data preprocessing
        self.train_data = self._sns_data_processing(self.train_data)

        # text vectorize and encode
        self.train_data = self._vectorize_and_encode(self.train_data, is_train=True)
        self.test_data = self._vectorize_and_encode(self.test_data, is_train=False)
        
        # split input, target data
        self.X_train, self.y_train = self._split_data(self.train_data)
        self.X_test, self.y_test = self._split_data(self.test_data)

    def _load_data(self, option):
        if option == 'train':
            data = pd.read_csv('data/train.csv')
        else:
            data = pd.read_csv('data/test.csv')
        return data
    
    def _sns_data_processing(self, data):
        """
        ### Preprocessing ###
        Perform preprocessing as needed *ex) Remove stop words, remove special characters, unify case, remove missing values, remove sentences that are too short
        Preprocess the text in data['content'] and store it in the data['content'] field *ex) data['content'] = data['content'].apply(lambda x: x.lower())
        *** All other fields can be used as training data except 'sentiment' field ***
        """
        ############################################## EDIT ################################################
        
        nltk.download('stopwords')
        nltk.download('punkt')

        #missing value
        data = data.dropna(subset=['content'])
        
        #capital delete
        data['content'] = data['content'].apply(lambda x: x.lower())
        
        #special characters
        data['content'] = data['content'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        #stop words
        stopWordSet = set(stopwords.words('english'))
        data['content'] = data['content'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stopWordSet]))
        
        #tokenize and length limit
        data = data[data['content'].apply(lambda x: len(word_tokenize(x)) >= 3)]
        
        ####################################################################################################
        return data

    def _vectorize_and_encode(self, data, is_train):
        data = self._vectorize_text(data,is_train)
        data = self._label_encoding(data)
        return data

    def _vectorize_text(self, data, is_train):
        """
        ### Vectorization ###
        Store the result of vectorizing text (content) in the data['vector'] field to vector using a technique such as one-hot vector or tf-idf and return it
        Below is an example code that shows the basics of vectorization: converting to a one-hot vector & generating vocab based on frequency of occurrence.
        """
        ############################################## EDIT ################################################
        
        if is_train:
            words = [word for text in data['content'] for word in text.lower().split()]
            wordCnt = Counter(words)
            min_frequency = 25
            vocab = {word for word, count in wordCnt.items() if count >= min_frequency}
            self.vocab = vocab
            print("vocab length: ", len(vocab))
            
            #document frequency for each word
            docFreq = {word: 0 for word in vocab}
            for text in data['content']:
                inText = set(text.lower().split())
                for word in inText:
                    if word in docFreq:
                        docFreq[word] += 1
            
            totalDocs = len(data['content'])
            self.idf = {word: math.log(totalDocs / (1 + docFreq[word])) for word in vocab}

        vocabDict = {word: idx for idx, word in enumerate(self.vocab)}
        # tfidfVectors = np.zeros((len(data['content']), len(vocab_index)))
        tfidfVectorList = []
        
        for text in data['content']:
            wordList = text.lower().split()
            wordCnt = Counter(wordList)
            lenDocs = len(wordList)
            
            tfidfVectorCache = [0.0 for _ in range(len(vocabDict))]

            for word, count in wordCnt.items():
                if word in vocabDict:
                    #TF Value
                    tf = count / lenDocs
                    idf = self.idf[word]
                    
                    tfidfVectorCache[vocabDict[word]] = tf * idf
            tfidfVectorList.append(tfidfVectorCache)

        data['vector'] = tfidfVectorList

        ####################################################################################################
        return data
    
    def _label_encoding(self, data):
        """
        ### Label Encoding ###
        Target is Positive, Neutral, Negative, and Irrelevant, Computers can't understand this natural language, so it needs to be converted to numbers.
        Convert Positive, Neutral, Negative, and Irrelevant to 0, 1, 2, and 3, respectively
        Convert the labels in data['sentiment'] to numbers and store them in the data['sentiment'] field
        """
        ############################################## EDIT ################################################
        
        numList = list()
        for sentiment in data['sentiment']:
            if sentiment == "Positive":
                numList.append(0)
            elif sentiment == "Neutral":
                numList.append(1)
            elif sentiment == "Negative":
                numList.append(2)
            else:
                numList.append(3)
                
        data['sentiment'] = numList
            
        ####################################################################################################
        return data
    
    def _split_data(self, data):
        X = np.array(data['vector'].tolist())
        y = data['sentiment'].to_numpy()
        y = y.reshape(len(y),)
        
        return X,y