import streamlit as st
import tensorflow as tf
import numpy as np
import re
import os
import pickle
import time

# --- Core Model and Preprocessing Logic (from your translate.py) ---
# It's best practice to keep these definitions in the app script.

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
        output, state = self.gru(x, initial_state=hidden)
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
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
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
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
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

# --- Streamlit Caching and Model Loading ---
# Use st.cache_resource to load the model and tokenizers only once.
@st.cache_resource
def load_model_and_tokenizers():
    # --- Parameters (must match your training script) ---
    embedding_dim = 256
    units = 1024
    
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
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    # Build the model by calling it once on dummy data
    max_length_inp = 11 # Adjust if your training parameters were different
    dummy_input = tf.random.uniform(shape=(1, max_length_inp), maxval=10, dtype=tf.int32)
    dummy_hidden = encoder.initialize_hidden_state()
    enc_out, _ = encoder(dummy_input, dummy_hidden)
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    _ = decoder(dec_input, dummy_hidden, enc_out)

    # Restore the latest checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    return encoder, decoder, inp_lang, targ_lang

# --- Translation Function ---
def evaluate(sentence, encoder, decoder, inp_lang, targ_lang):
    # --- Parameters (must match your training script) ---
    units = 1024
    max_length_inp = 11 # Adjust if different
    max_length_targ = 17 # Adjust if different

    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index.get(i, 0) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = targ_lang.index_word.get(predicted_id, '')
        
        if word == '<end>':
            return result
        
        result += word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return result

# --- Streamlit UI Layout ---

st.set_page_config(layout="centered")

st.title("German to English Neural Machine Translation 🇩🇪 -> 🇬🇧")
st.write("This app uses a sequence-to-sequence model with Bahdanau attention to translate German sentences into English.")

# Load the model and show a spinner
with st.spinner('Loading the translation model, please wait...'):
    encoder, decoder, inp_lang, targ_lang = load_model_and_tokenizers()

# Get user input
german_text = st.text_area("Enter a German sentence:", "ich möchte ein bier.", height=100)

if st.button("Translate", type="primary"):
    if german_text:
        with st.spinner('Translating...'):
            time.sleep(1) # Small delay for better UX
            translated_text = evaluate(german_text, encoder, decoder, inp_lang, targ_lang)
            st.subheader("Predicted Translation")
            st.success(translated_text.strip())
    else:
        st.warning("Please enter a sentence to translate.")

st.markdown("---")
st.write("Built by a Coding Partner.")