import re
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding

config = ConfigProto()
config.gpu_options.allow_growth=True
session = InteractiveSession(config=config)

num_examples=30000
data = pd.read_csv('fra-eng/fra.txt', sep='\t', header=None)

data = data.iloc[:, 0:2]

data = data.iloc[:num_examples, :]

def preprocess_sentences(sent):
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)
    sent = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sent)
    sent = sent.strip()
    sent = '<start> '+sent+' <end>'
    return sent

def max_length(lang):
  return max(len(t) for t in lang)

def tokenize(lang):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(lang)
    sequence = tokenizer.texts_to_sequences(lang)
    seqeunces = pad_sequences(sequence, padding='post')
    return seqeunces, tokenizer

def load_dataset(data):
    inp = [preprocess_sentences(sent) for sent in data.iloc[:, 1]]
    tar = [preprocess_sentences(sent) for sent in data.iloc[:, 0]]
    
    inp, inp_tokenizer = tokenize(inp)
    tar, tar_tokenizer = tokenize(tar)
    
    return inp, tar, inp_tokenizer, tar_tokenizer

inp, tar, inp_tokenizer, tar_tokenizer = load_dataset(data)

max_length_inp, max_length_tar = max_length(inp), max_length(tar)

inp_train, inp_val, tar_train, tar_val = train_test_split(inp, tar, test_size=0.2)

buffer_size = len(inp_train)
batch_size = 64
steps_per_epoch = len(inp_train)//batch_size
output_dim = 256
units = 1024
vocab_inp_size = len(inp_tokenizer.word_index)+1
vocab_tar_size = len(tar_tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((inp_train, tar_train)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

class Encoder(tf.keras.models.Model):
    
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size, max_length_inp):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size,embedding_dim, input_length=max_length_inp)
        self.lstm = LSTM(self.enc_units, input_shape=(max_length_inp, embedding_dim), 
                         return_sequences=True,
                         return_state=True,
                         recurrent_initializer='glorot_uniform')
    
    def call(self, x, hidden_state):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=[hidden_state, hidden_state])
        return output, [state_h, state_c]
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

encoder = Encoder(vocab_inp_size, output_dim, units, batch_size, max_length_inp)

class Decoder(tf.keras.models.Model):
    
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, max_length_tar):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size,embedding_dim, input_length=max_length_tar)
        self.lstm = LSTM(self.dec_units, input_shape=(max_length_tar, embedding_dim), 
                         return_sequences=True,
                         return_state=True)
        self.fc = Dense(vocab_size)
    
    def call(self, x, hidden_state, encoder_output):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, [state_h, state_c]

decoder = Decoder(vocab_tar_size, output_dim, units, batch_size, max_length_tar)

optimizer = tf.keras.optimizers.Adam()
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

@tf.function
def train_step(inp, tar, enc_hidden):
    loss = 0
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tar_tokenizer.word_index['<start>']] * batch_size, 1)
        for t in range(1, tar.shape[1]):
            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
            loss += tf.reduce_mean(loss_obj(tar[:, t], predictions))
            dec_input = tf.expand_dims(tar[:, t], 1)
    batch_loss = loss/(int(tar.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradient = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradient, variables))
    return batch_loss

epochs = 10
for epoch in range(epochs):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    loss = 0
    
    for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, tar,enc_hidden)
        loss += batch_loss
        
        if batch % 100 == 0:
            print("Epoch: {}, Batch: {}, Loss:{}".format(epoch+1, batch, batch_loss.numpy()))
    if (epoch+1)%2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    print('Epoch: {} Loss: {}'.format(epoch, loss/steps_per_epoch))
    print('Time taken for 1 eopch {}'.format(time.time() - start))

def evaluate(sentence):
    sentence = preprocess_sentences(sentence)
    inputs = [inp_tokenizer.word_index[word] for word in sentence.split(' ')]
    inputs = pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_output, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tar_tokenizer.word_index['<start>']], 0)
    
    for i in range(max_length_tar):
        prediction, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
        predicted_id = tf.argmax(prediction[0]).numpy()
        result += tar_tokenizer.index_word[predicted_id] + ' '
        
        if tar_tokenizer[predicted_id] == '<end>':
            return result, sentence
        
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence

def translate(sentence):
    result, sentence = evaluate(sentence)
    
    print('Input: %s' %(sentence))
    print('Predicted Sentence: %s' %(result))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'hace mucho frio aqui.')

translate(u'esta es mi vida.')
