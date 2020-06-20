# neural-machine-translation

This notebook trains a sequence to sequence (seq2seq) model for French to English translation. This is an advanced example that assumes some knowledge of sequence to sequence models.

After training the model in this notebook, you will be able to input a French sentence and return the English translation.
# Write the encoder and decoder model

Implement an encoder-decoder model with attention which you can read about in the TensorFlow Neural Machine Translation (seq2seq) tutorial. This example uses a more recent set of APIs. This notebook implements the attention equations from the seq2seq tutorial. The following diagram shows that each input words is assigned a weight by the attention mechanism which is then used by the decoder to predict the next word in the sentence. The below picture and formulas are an example of attention mechanism from Luong's paper.

The input is put through an encoder model which gives us the encoder output of shape (batch_size, max_length, hidden_size) and the encoder hidden state of shape (batch_size, hidden_size).

This tutorial uses Bahdanau attention for the encoder. Let's decide on notation before writing the simplified form:

FC = Fully connected (dense) layer  
EO = Encoder output  
H = hidden state  
X = input to the decoder  

And the pseudo-code:

score = FC(tanh(FC(EO) + FC(H)))  
attention weights = softmax(score, axis = 1). Softmax by default is applied on the last axis but here we want to apply it on the 1st axis, since the shape of score is (batch_size, max_length, hidden_size). Max_length is the length of our input. Since we are trying to assign a weight to each input, softmax should be applied on that axis.  
context vector = sum(attention weights * EO, axis = 1). Same reason as above for choosing axis as 1.  
embedding output = The input to the decoder X is passed through an embedding layer.  
merged vector = concat(embedding output, context vector)  

This merged vector is then given to the GRU

# Training

Pass the input through the encoder which return encoder output and the encoder hidden state.  
The encoder output, encoder hidden state and the decoder input (which is the start token) is passed to the decoder.  
The decoder returns the predictions and the decoder hidden state.  
The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.  
Use teacher forcing to decide the next input to the decoder.  
Teacher forcing is the technique where the target word is passed as the next input to the decoder.  
The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
