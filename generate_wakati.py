# coding: utf-8

import sys
import MeCab


def get_std_out(path, lab=None):
    txt_list = read_text(path)
    out_str = ""
    for line in txt_list:
        if not lab:
            out_str += " ".join(line) + "\n"
        else:
            out_str += lab + ","+ " ".join(line) + "\n"
    return out_str

def read_text(path):
    with open(path, mode='rb') as f:
        txt_list = []
        for line in f:
            txt_list.append(split_sentence(line))
    return txt_list

def split_sentence(text):
    tagger = MeCab.Tagger("-Owakati")
    text_sp = tagger.parse(text)
    sp_list = text_sp.replace("\n","").split(" ")
    sp_list.remove("")
    return sp_list

def write_wakati(path, out_str):
    with open(path, mode="wb") as f:
        f.write(out_str)

if __name__=="__main__":
    path_genju = "./data/text/genju.txt"
    path_yakiu = "./data/text/yakiu.txt"
    path_jawiki = "./data/text/jawiki.txt"
    # jawiki
    out_str = get_std_out(path_jawiki)
    out_path = "./data/text/wakati.txt"
    write_wakati(out_path, out_str)
    # やきう民
    out_str = get_std_out(path_yakiu, "yakiu")
    out_path = "./data/text/yakiu_w.csv"
    write_wakati(out_path, out_str)
    # 原住民
    out_str = get_std_out(path_genju, "genju")
    out_path = "./data/text/genju_w.csv"
    write_wakati(out_path, out_str)
