import proyecto

WORD2VEC = "word2vec"
FASTTEXT = "fasttext"
GLOVE = "glove"

vectors = proyecto.execute_Models([WORD2VEC],"/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional/Textos",100,5,0,"/media/david/24ACA5E9ACA5B628/docs/3er_ano/Practica_profesional/Vectores")

tokens = proyecto._tokenize_text_list(["La niña fue al parque. Ella y Juan están en la mesa, junto a él."])
print("Vectors\n")

proyecto.plot_3d_vectors(vectors)

