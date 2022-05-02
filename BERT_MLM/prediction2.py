from pip import main
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI,Body, Request
from pydantic import BaseModel
from loguru import logger
import numpy as np
from typing import List
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import load_model

import dataprep
app = FastAPI()
word_tokenizer, id2token, token2id = dataprep.tokenizer()
mask_token_id = word_tokenizer(["[mask]"]).numpy()[0][0]

att_model =tf.keras.models.load_model('bert_mlm_model.h5')

def mask_filler(x):
  # x = (['I have wathed this [mask] and it was awesome'])
    pred_tokens = word_tokenizer(x)  #### word_tokenizer class
  # print(pred_tokens)
    pred_vectors = att_model.predict(pred_tokens)
  # print(pred_tokens)
    mask_position = np.where(pred_tokens == mask_token_id)[1]
  # print(mask_position)
    pred_vectors =  tf.squeeze(pred_vectors)
  # print(pred_vectors.shape)
    k = 5
    max_probs = tf.math.top_k(pred_vectors[4],k =5)[0]
    max_positions = tf.math.top_k(pred_vectors[4],k =5)[1]
  # print(max_positions)
    # for i in range(k):
    #   print('PREDICTED_TOKEN:',id2token[int(max_positions[i])],'  ',
    #       'PREDICTION_PROBABILITY',np.array(max_probs[i]))
  
    return (id2token[int(max_positions[0])]) 
# text = ['I have wathed this [mask] and it was awesome']
# score = mask_filler(text)
# print('MASK_TOKEN',score)
@app.get('/prediction')
def fetch_predictions(text:str):
    res = mask_filler([text])
    return {'mask' : res}
#     score = mask_filler(text)
    
#     return score

if __name__ == '__main__':
    # main()
    uvicorn.run(app)