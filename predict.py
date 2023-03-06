import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdown

import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer, TFBertModel

MAX_LEN = 200
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SeqWeightedAttention(tf.keras.layers.Layer):
    r"""Y = \text{softmax}(XW + b) X
    See: https://arxiv.org/pdf/1708.00524.pdf
    """

    def __init__(self, use_bias=True, return_attention=False, **kwargs):
        super(SeqWeightedAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.return_attention = return_attention
        self.W, self.b = None, None

    def get_config(self):
        config = {
            'use_bias': self.use_bias,
            'return_attention': self.return_attention,
        }
        base_config = super(SeqWeightedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.W = self.add_weight(shape=(int(input_shape[2]), 1),
                                 name='{}_W'.format(self.name),
                                 initializer=tf.keras.initializers.get('uniform'))
        if self.use_bias:
            self.b = self.add_weight(shape=(1,),
                                     name='{}_b'.format(self.name),
                                     initializer=tf.keras.initializers.get('zeros'))
        super(SeqWeightedAttention, self).build(input_shape)

    def call(self, x, mask=None):
        logits = tf.keras.backend.dot(x, self.W)
        if self.use_bias:
            logits += self.b
        x_shape = tf.keras.backend.shape(x)
        logits = tf.keras.backend.reshape(logits, (x_shape[0], x_shape[1]))
        ai = tf.keras.backend.exp(logits - tf.keras.backend.max(logits, axis=-1, keepdims=True))
        if mask is not None:
            mask = tf.keras.backend.cast(mask, tf.keras.backend.floatx())
            ai = ai * mask
        att_weights = ai / (tf.keras.backend.sum(ai, axis=1, keepdims=True) + tf.keras.backend.epsilon())
        weighted_input = x * tf.keras.backend.expand_dims(att_weights)
        result = tf.keras.backend.sum(weighted_input, axis=1)
        #if self.return_attention:
        #    return att_weights
        return result, att_weights, weighted_input

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return input_shape[0], output_len

    def compute_mask(self, _, input_mask=None):
        if self.return_attention:
            return [None, None]
        return None

    @staticmethod
    def get_custom_objects():
        return {'SeqWeightedAttention': SeqWeightedAttention}
    

def create_model():

  ## Inputs
  input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name="input_ids", dtype=tf.int32)
  attention_masks = tf.keras.layers.Input(shape=(MAX_LEN,), name="attention_masks", dtype=tf.int32)
  token_type_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name="token_type_ids", dtype=tf.int32)

  # BERT encoder
  encoder = TFBertModel.from_pretrained("bert-base-uncased") 
  embedding = encoder(
      input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks
  )[0]


  [attn, attn_w, weighted_input] = SeqWeightedAttention(
                                          return_attention=True)(embedding, mask=attention_masks)

  midLayer=keras.layers.Dropout(0.2)(attn)

  # midLayer=keras.layers.Dense(400,activation='elu',kernel_initializer="he_normal")(midLayer)
  midLayer=keras.layers.BatchNormalization()(midLayer)

  ## Classification Output Layer
  probabilities = keras.layers.Dense(6, activation='softmax')(midLayer)


  ## Model
  model = keras.Model(
      inputs=[input_ids, attention_masks, token_type_ids],
      outputs=[probabilities],
  )
  loss="categorical_crossentropy"
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
  model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])

  return model

url='https://drive.google.com/file/d/1qLntWddYtP2k5KiZiDqDDSWW2uNn8EP0/view?usp=share_link'
output='models/model.h5'
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

emotionsModel=create_model()
emotionsModel.load_weights('models/model.h5')

def convertSentence(text,tokenizer=tokenizer ,max_seq_length=MAX_LEN):  
  input_ids,attention_masks,token_type_ids =[], [], []
  i=text
  input_dict = tokenizer.encode_plus(
          i,
          add_special_tokens=True,
          max_length=max_seq_length, # truncates if len(s) > max_length
          return_token_type_ids=True,
          return_attention_mask=True,
          padding='max_length', # pads to the right by default # CHECK THIS for pad_to_max_length
          truncation=True
      )
  in_id=input_dict["input_ids"]
  input_ids.append(in_id)
  attention_mask=input_dict['attention_mask']
  attention_masks.append(attention_mask)
  token_type_id=input_dict["token_type_ids"]
  token_type_ids.append(token_type_id)
      
     
  return (
    np.array(input_ids),
    np.array(attention_masks),
    np.array(token_type_ids),
  )

def predictEmotion(sen,model=emotionsModel):
  sen = " ".join(sen.split())
  (sen_input_ids, sen_attention_masks, sen_token_type_ids)=convertSentence(sen)

  emotionsList=['anger', 'love', 'fear', 'joy', 'sadness', 'surprise']

  preds=model.predict([sen_input_ids, sen_attention_masks, sen_token_type_ids])
  emotionIndex=np.where(preds[0]==preds[0].max())[0][0]

  return emotionsList[emotionIndex]
