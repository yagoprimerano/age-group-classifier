# install the tagger with "python -m spacy download pt_core_news_lg"

import numpy as np
from tensorflow import keras
import spacy
from spacy.lang.pt.examples import sentences 
import pandas as pd



################ Data Path #################################
data_input_path = "../../data/raw/(CORRIGIDO)ep2_pln_train.xlsx"
txt_column = "req_text"

data_output_path = "../../data/processed/(CORRIGIDO)ep2_pln_train_pos_pt_core_news_lg.xlsx"
############################################################



data = pd.read_excel(data_input_path)

pos_converter = spacy.load("pt_core_news_lg")
pos_texts = [pos_converter(text) for text in data[txt_column]]

pos = np.empty(len(pos_texts), dtype='object')

pos = np.empty(len(pos_texts), dtype='object')
for i in range(len(pos_texts)):
    pos[i] = " ".join([token.pos_ for token in pos_texts[i]])
    
    
data["pos"] = pos


data.to_csv(data_output_path)