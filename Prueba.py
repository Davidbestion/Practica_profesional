#Prueba

import proyecto
import os

WORD2VEC = "word2vec"
FASTTEXT = "fasttext"
GLOVE = "glove"

vectors = proyecto.Execute_Models([WORD2VEC,FASTTEXT,GLOVE],"/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional/Textos",100,5,0,"/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional/Vectores")

tokens = proyecto._tokenize_text_list(["La niña fue al parque. Ella y Juan están en la mesa, junto a él."])
print("Vectors\n")

#ESTO NO PINCHA...
# for model in vectors:
#    for element in model:
#       print(element + " ")
#       for v in element[1]:
#          print(v + " ")
#       print("\n")
#    print("\n")


