import os
import numpy as np
import subprocess

#import BERTmaster.run_pretraining

#from transformers import BertModel, BertTokenizer, BertForMaskedLM
#import transformers
#import BERTmaster.extract_features as Bert

from spacy.lang.es import Spanish

import gensim.models as models
   
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
            with open(os.path.join(path, filename), 'r', encoding='utf-8', errors='ignore') as f:
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

def Execute_Models(models :list[str], load_path :str, size :int, window :int, min_count :int, save_path :str):
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
    texts = ReadTexts(load_path)
    tokens = tokenize_text_list(texts)
    
    for func_name in models: 
        func = globals().get(func_name) #get the functions which names are in the "models" list
        if func and callable(func):
            results.append(func(tokens, size, window, min_count, save_path)) #store the results of the functions.
        else: raise ValueError("Error getting function in 'Execute_Models' function. Please check the list of models.")
    return results
    
def Word2Vec(token_list :list, size :int, window :int, min_count :int, path :str):
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
    model = models.Word2Vec(sentences=token_list, vector_size=size, window=window, min_count=min_count)
    vectors = model.wv.vectors

    Save_Vectors(vectors, path, "Word2Vec_vectors.txt")
    return vectors
    
def FastText(token_list :list, size :int, window :int, min_count :int, path :str):
    """Train a FastText model.

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
    model = models.FastText(sentences=token_list, vector_size=size, window=window, min_count=min_count)
    vectors = model.wv.vectors

    Save_Vectors(vectors, path, "FastText_vectors.txt")

    return vectors

#def BERT(token_list):#ATENCION NO SE SI ESTA BIEN.
    #model = BERTmaster
    #model = transformers.

def GloVe(texts: list, size :int, window :int, min_count :int, path :str):
    with open("texts.txt", "w") as f: #HACER ESTO PARA LOS DEMAS MODELOS
        for text in texts:            #EXTRAER LOS TEXTOS NO DE LOS ARCHIVOS ORIGINALES SINO DE ESTE
            for word in text: 
                f.write(word + ' ')
            f.write("\n")
        
    corpus="texts.txt"
    vocab_file="GloVeMaster/GloVeMaster/vocab.txt"
    coocurrence_file="GloVeMaster/GloVeMaster/cooccurrence.bin"
    coocurrence_shuf_file="GloVeMaster/GloVeMaster/cooccurrence.shuf.bin"
    builddir="GloVeMaster/GloVeMaster/build"
    save_file= path + "/GloVe_vectors"
    verbose=2
    memory=4.0
    vocab_min_count= str(min_count)
    vector_size= str(size)
    max_iter=15
    window_size= str(window)
    binary=2
    num_threads=8
    x_max=10

    output = subprocess.check_output("echo", shell=True)
    print(output.decode())
    #                                  $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
    output = subprocess.check_output(builddir +"/vocab_count -min-count "+str(vocab_min_count)+ " -verbose "+str(verbose)+" < "+corpus+" > "+vocab_file, shell=True)
    print(output.decode())
    #                                  $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
    output = subprocess.check_output(builddir +"/cooccur -memory "+str(memory)+" -vocab-file "+vocab_file+" -verbose "+str(verbose)+" -window-size "+str(window_size)+" < "+corpus+" > "+coocurrence_file, shell=True)
    print(output.decode())
    #                                  $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    output = subprocess.check_output(builddir +"/shuffle -memory "+str(memory)+" -verbose "+str(verbose)+" < "+coocurrence_file+" > "+coocurrence_shuf_file, shell=True)
    print(output.decode())
    #                                  $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
    output = subprocess.check_output(builddir +"/glove -save-file "+save_file+" -threads "+str(num_threads)+" -input-file " + coocurrence_shuf_file + " -x-max "+str(x_max)+"-iter "+str(max_iter)+" -vector-size "+str(vector_size)+" -binary "+str(binary)+" -vocab-file "+vocab_file+" -verbose "+str(verbose), shell=True)
    print(output.decode())

    #Esto es pa retornar los vectores en el mismo formato que los metodos anteriores
    with open(save_file + ".txt", "r") as f:
        lines = f.readlines()

    vectors = []
    #Los otros metodos devuelven un array con los vectores, 
    # que tambien son arrays con las respectivas componentes.
    for l in lines:
        numbers = [str(x) for x in l.strip().split(' ')]
        #ACLARACION: quito el primer elemento para que lo devuelto sean solo los vectores.
        #Este metodo de GloVe devuelve delante de las componentes, la palabra correspondiente a cada vector.
        numbers = [float(x) for x in numbers[1:]]
        vectors.append(numbers)
    
    vectors = np.array(vectors)
    return vectors



def Save_Vectors(vectors: list[list], path, file_name):
    
    with open(os.path.join(path,file_name), "w") as f:
        for vector in vectors:
            for number in vector:
                f.write(str(number) + " ")
            f.write("\n")


