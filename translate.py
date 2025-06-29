import tensorflow as tf
import numpy as np
import re
import os
import pickle

# --- Essential Definitions from your Training Script ---
# (Your class definitions for Encoder, BahdanauAttention, and Decoder go here, no changes needed)
def preprocess_sentence(w):
  w = w.lower().strip()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.strip()
  w = '<start> ' + w + ' <end>'
  return w

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)
    return x, state, attention_weights


# --- Loading and Translation Logic ---

# --- Important Parameters ---
embedding_dim = 256
units = 1024
max_length_inp = 11  # Adjust if you changed your training parameters
max_length_targ = 17 # Adjust if you changed your training parameters

# Load the tokenizers
with open('inp_lang.pickle', 'rb') as handle:
    inp_lang = pickle.load(handle)

with open('targ_lang.pickle', 'rb') as handle:
    targ_lang = pickle.load(handle)

vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

# Initialize the model components
encoder = Encoder(vocab_inp_size, embedding_dim, units, 1)
decoder = Decoder(vocab_tar_size, embedding_dim, units, 1)
optimizer = tf.keras.optimizers.Adam()

# Point to the checkpoint directory
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# ==========================================================
# V V V V V V V V   THE FIX IS HERE   V V V V V V V V V V V V
# ==========================================================
# To fix the "unbuilt state" warning, we must "build" the model by calling it once
# on some dummy data before restoring the checkpoint.

# Create some dummy inputs with the correct shapes
dummy_input = tf.random.uniform(shape=(1, max_length_inp), maxval=10, dtype=tf.int32)
dummy_hidden = encoder.initialize_hidden_state()
enc_out, enc_hidden = encoder(dummy_input, dummy_hidden)
dec_hidden = enc_hidden
dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

# Call the decoder once to build it as well
_ = decoder(dec_input, dec_hidden, enc_out)

# ==========================================================
# ^ ^ ^ ^ ^ ^ ^ ^ ^  END OF THE FIX  ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
# ==========================================================


# NOW, restore the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
print("Model restored successfully.")


# --- (Your evaluate and translate functions go here, no changes needed) ---
def evaluate(sentence):
  sentence = preprocess_sentence(sentence)
  inputs = [inp_lang.word_index.get(i, 0) for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)
  result = ''
  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)
  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
  for t in range(max_length_targ):
    predictions, dec_hidden, _ = decoder(dec_input,
                                         dec_hidden,
                                         enc_out)
    predicted_id = tf.argmax(predictions[0]).numpy()
    result += targ_lang.index_word.get(predicted_id, '') + ' '
    if targ_lang.index_word.get(predicted_id, '') == '<end>':
      return result, sentence
    dec_input = tf.expand_dims([predicted_id], 0)
  return result, sentence

def translate(sentence):
  result, sentence = evaluate(sentence)
  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))
  print(f"Clean translation: {result.replace('<start>', '').replace('<end>', '').strip()}")


# --- Now you can translate any sentence you want! ---
translate(u'Bitte.')