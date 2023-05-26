import os
#from transformers import BertModel, BertTokenizer, BertForMaskedLM
#import transformers
import BERTmaster.modeling as Bert

from spacy.lang.es import Spanish

import gensim.models.poincare as poinc
import gensim.models as models

import gensim_old.models as old


    
def ReadTexts(path : str):
    """Extracts the text of the .txt files in the given path.
    
    Parameters:
    path - path to the files

    Returns:
    doc_list - list of the texts stored in the files of the given path. Each element of the list is the text of a different file.
    """
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

def Execute_Models(models :list[str], load_path :str, size :int, window :int, min_count :int, save_path :str, mode :int=0):
    """Trains all the modles requested in "models".

    Parameters:
    models: list of models' names.
    load_path: path to the files containing the texts.
    size: length of the vectors resulting from training the model.
    window: maximum distance between the current and predicted word within a sentence.
    min_count: if a word appears less than this number of times then it will be ignored.
    save_path: path to a file to save the vectors.
    mode: number = {-1, 0, 1}. Defines the return mode.

    Return:
    results: list containing lists with the vectors retorned by the different models.
    """
    results = []
    tokens = tokenize_text_list(ReadTexts(load_path))
    
    for func_name in models: 
        func = globals().get(func_name) #get the functions which names are in the "models" list
        if func and callable(func):
            results.append(func(tokens, size, window, min_count, save_path, mode)) #store the results of the functions.
        else: raise ValueError("Error getting function in 'Execute_Models' function. Please check the list of models.")
    return results
    
def Word2Vec(token_list :list, size :int, window :int, min_count :int, path :str, mode :int):
    """Train a Word2Vec model.

    Parameters:
    token_list: list of lists of the tokenized texts. Each element of the bigger list is a list, and every inner list is a list of tokens, a tokenized text.
    size: length of the vectors resulting from training the model.
    window: maximum distance between the current and predicted word within a sentence.
    min_count: if a word appears less than this number of times then it will be ignored.
    path: path to a file to save the vectors.
    mode: number = {-1, 0, 1}. Defines the return mode.
   
    Returns:
    If mode = -1, the function returns the vectors.
    If mode = 1, the function does not return the vectors but save them in a file at the giving path.
    If mode = 0, do both.
    """
    model = models.Word2Vec(token_list, size, window, min_count, workers=4)
    vectors = model.wv.vectors

    if mode == 1: #Save the vectors in a file located at path
        vectors.save(path)
        return 
    if mode == -1: #Returns the vectors.
        return vectors
    else: #Do both.
        vectors.save(path)
        return vectors
    
def FastText(token_list :list, size :int, window :int, min_count :int, path :str, mode :int):
    model = models.FastText(token_list, size=100, window=5, min_count=1, workers=4)
    vectors = model.wv.vectors

    if mode == 1: #Save the vectors in a file located at path
        vectors.save(path)
        return 
    if mode == -1: #Returns the vectors.
        return vectors
    else: #Do both.
        vectors.save(path)
        return vectors
    
def Poincare(token_list):
    model = poinc.PoincareModel(token_list, size=100, window=5, min_count=1, workers=4)
    relations = poinc.po
    return model

def BERT(token_list):#ATENCION NO SE SI ESTA BIEN.
    modelConfig = Bert.BertConfig()
    model = Bert.BertModel()
    #model = transformers.

def GloVe(token_list):
    model = glove.Glove()
