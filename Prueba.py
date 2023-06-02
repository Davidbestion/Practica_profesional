#Prueba

import proyecto
import os

vectors = proyecto.Execute_Models(["Word2Vec","FastText","GloVe"],"/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional/Textos",100,5,0,"/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional/Vectores")

print("Vectors\n")

for v in vectors:
   print(v)
