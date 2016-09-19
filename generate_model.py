# coding: utf-8

from gensim.models import word2vec

def generate_w2v_model(path, vec_dim=200):
    data = word2vec.Text8Corpus(path)
    model = word2vec.Word2Vec(data, size=vec_dim)
    return model

if __name__=="__main__":
    path = "./data/text/wakati.txt"
    model = generate_w2v_model(path)
    model_path = "./data/model/jawiki.model"
    model.save(model_path)

