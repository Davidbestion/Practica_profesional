import gensim as gen
import os
from spacy.lang.es import Spanish
from spacy.tokenizer import Tokenizer

#import gensim.models.word2vec as w2v
#import gensim.models.fasttext as fastt
#import gensim.models.poincare as poinc
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
      
    def TokenizeText(Text):#Text es el texto entero de un documento.
        nlp = Spanish()
        tokenizer = nlp.tokenizer
        tokens = tokenizer(Text) 
        return tokens
    



#Method that processes a text and returns a list of the vectors of each word using word2vec
def word2vec(text):
    model = gen.models.Word2Vec(text, size, window, min_count, workers, iter)
    