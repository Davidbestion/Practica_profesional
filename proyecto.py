import gensim as gen
from spacy.lang.es import Spanish
from spacy.tokenizer import Tokenizer

#import gensim.models.word2vec as w2v
#import gensim.models.fasttext as fastt
#import gensim.models.poincare as poinc
#import gensim.scripts.glove2word2vec as glove #esto transforma de GloVe a word2vec

nlp = Spanish()
tokenizer = nlp.tokenizer
tokens = tokenizer(text) 


#Method that processes a text and returns a list of the vectors of each word using word2vec
def word2vec(text):
    model = gen.models.Word2Vec(text, size, window, min_count, workers, iter)
    