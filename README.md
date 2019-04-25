# Movie-Data-NER
Bi-directional LSTM for Sequence Labelling utilizing MIT Movie Dataset

The MIT Movie Dataset can be found here: https://groups.csail.mit.edu/sls/downloads/movie/

Model structure:
  - Embed words using pre-trained GloVe embeddings
  - At each word, run character level bi-LSTM. Select the last hidden state and concatenate to relevant word embedding
  - Bi-directional LSTM utilizing word embedding + char Bi-LSTM hidden states

Next Steps: add an additional CRF layer which would be decoded via a Viterbi loss function
