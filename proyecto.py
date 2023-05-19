import gensim as gen
from spacy.lang.es import Spanish
from spacy.tokenizer import Tokenizer

import gensim.models.word2vec as word2vec
import gensim.models.poincare as poinc
import gensim.models as models

import zipimport
lib_zip = zipimport.zipimporter('BERT-master.zip')
BERT = lib_zip.load_module('lib.BERT')


#import gensim.scripts.glove2word2vec as glove #esto transforma de GloVe a word2vec

class WordEmbedding(object):
    #Method that reads text from docs and returns a list with the text of each file.
    def ReadTexts():
        path = "path/to/directory" # change to the directory path where your text files are stored
        doc_list = []
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                with open(os.path.join(path, filename), 'r') as f:
                    text = f.read().replace('\n', ' ')
                    doc_list.append(text)
        return doc_list
      
    def tokenize_text_list(text_list):
        nlp = Spanish()
        tokenizer = nlp.tokenizer
        token_list = []
        for text in text_list:
            tokens = tokenizer(text)
            token_list.append([token.text for token in tokens]) #token_list.append(tokens)
        return token_list
    
    def Word2Vec(token_list):
        model = models.Word2Vec(token_list, size=100, window=5, min_count=1, workers=4)
        return model
    
    def FastText(token_list):
        model = models.FastText(token_list, size=100, window=5, min_count=1, workers=4)
        return model
    
    def Poincare(token_list):
        model = poinc.PoincareModel(token_list, size=100, window=5, min_count=1, workers=4)
        return model
    
    def GloVe(token_list):
        model = gen.utils.

    def BERT(token_list):
        
    

    