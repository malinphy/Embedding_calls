import pandas as pd 
import numpy as np 
import tensorflow as tf 
print('tensorflow version',tf.__version__)
from platform import python_version
print('python_version:',python_version())
from tensorflow import keras 
from tensorflow.keras import layers 
# from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re,string, unicodedata

# class data_prep:
#   def __init__(self,data_url):
#     self.data_url = data_url

VOCAB_SIZE = 30000
max_seq = 256
MAX_LEN = 256

def tokenizer():
  data_url = 'https://raw.githubusercontent.com/malinphy/datasets/main/IMDB_sent/IMDB%20Dataset.csv'
  df = pd.read_csv(data_url)

  def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), "")

  def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

  # df['review'] = df['review'].apply(custom_standardization)
  df['review'] = df['review'].apply(strip_accents)
  
  def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
      """Build Text vectorization layer

     Args:
      texts (list): List of string i.e input texts
      vocab_size (int): vocab size
      max_seq (int): Maximum sequence lenght.
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
      vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        standardize=custom_standardization,
        output_sequence_length=max_seq,
        )
    
      vectorize_layer.adapt(texts)

      # Insert mask token in vocabulary
      vocab = vectorize_layer.get_vocabulary()
      vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[mask]"]
      vectorize_layer.set_vocabulary(vocab)
      return vectorize_layer

  vectorize_layer = get_vectorize_layer(df.review.values.tolist(),
                                          VOCAB_SIZE,
                                          MAX_LEN,
                                          special_tokens=["[mask]"],)


  id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
  token2id = {y: x for x, y in id2token.items()}

  return vectorize_layer, id2token, token2id


vectorize_layer, id2token, token2id = tokenizer()

print('HELLO')