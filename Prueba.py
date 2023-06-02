#Prueba

import proyecto
import os

vectors = proyecto.Execute_Models(["GloVe"],"/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional/Textos",100,5,0,"/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional/Vectores")

print("Vectors\n")

for v in vectors:
   print(v)
# path = "/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional"
# filename = "Word2Vec_vectors.txt"

# with open(os.path.join(path, filename), 'r', encoding='utf-8', errors='ignore') as f:
#     text = f.readlines()
#     print(text[0])