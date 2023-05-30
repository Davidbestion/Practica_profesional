#Prueba

import proyecto

vectors = proyecto.Execute_Models(["GloVe"],"F:/docs/3er_ano/Practica_profesional/Textos",100,5,0,"F:/docs/3er_ano/Practica_profesional/Vectores",0)

print("Vectors\n")

for v in vectors:
    print(v)