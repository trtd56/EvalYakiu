# -*- coding: utf-8 -*-

from gensim.models import word2vec
from generate_wakati import split_sentence

class TransVecotr():

    def __init__(self, model_path):
        self.model = word2vec.Word2Vec.load(model_path)
        self.high = 0
        self.width = 0

    def __call__(self, path):
        """
        param path str: wakati.txt path
        """
        txt_list, lab_list = self.read_txt(path)
        rev_doc_list, max_len = self.generate_rev_doc(txt_list)
        doc_vec_list, width = self.trans_vector(rev_doc_list)
        dataset = {"vec":doc_vec_list,"txt":rev_doc_list, "lab":lab_list}
        return dataset, max_len, width

    def padding(self, document_list, max_len):
        new_document_list = []
        for doc in document_list:
            pad_line = ['<pad>' for i in range(max_len - len(doc))]
            new_document_list.append(doc + pad_line)
        return new_document_list

    def read_txt(self, path):
        txt_list = []
        lab_list = []
        with open(path, mode='rb') as f:
            for line in f:
                lab, txt= line.split(",")
                txt_list.append(txt.split())
                lab_list.append(lab)
        return txt_list, lab_list

    def generate_rev_doc(self, txt_list):
        max_len = 0
        rev_doc_list = []
        for txt in txt_list:
            rev_doc = []
            for word in txt:
                try:
                    word_vec = self.model[word.decode('utf-8')]
                    rev_doc.append(word)
                except KeyError:
                    rev_doc.append('<unk>')
            rev_doc_list.append(rev_doc)
            if len(rev_doc) > max_len:
                max_len = len(rev_doc)
        rev_doc_list = self.padding(rev_doc_list, max_len)
        return rev_doc_list, max_len

    def trans_vector(self, rev_doc_list):
        width = 0
        doc_vec_list = []
        for doc in rev_doc_list:
            doc_vec = []
            for word in doc:
                try:
                    vec = self.model[word.decode('utf-8')]
                except KeyError:
                    vec = self.model.seeded_vector(word)
                doc_vec.append(vec)
                width = len(vec)
            doc_vec_list.append(doc_vec)
        return doc_vec_list, width

    def gen_pred_vec(self, text, max_len):
        sp_text = [split_sentence(text), [u"_" for i in range(max_len)]]
        rev_doc_list, _ = self.generate_rev_doc(sp_text)
        doc_vec_list, width = self.trans_vector(rev_doc_list)
        return doc_vec_list[0]
