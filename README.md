# Transformer-Neural-Machine-Translation
baseline Neural Machine Translation model for German to English

Includes LSTM, encoder-decoder with attention, and Transformer model architecture.


File description:
• train.py* is used to train the translation models.

• translate.py* translates the test-set greedily using model parameters restored from the best checkpoint file and saves the output to model translations.txt.

• example.sh is a suggested outline of a single experiment run to train a model, generate translations and then find the test-set BLEU score. To train a baseline model, follow example.sh without modifying any lines. This script includes training and inference. 

• After each epoch in training, the latest model file is saved to disk as checkpoint last.pt.  

• The translations will be output to the file model translations.txt.

• The multi-bleu.perl script is used to calculate the test-BLEU score of the baseline model.
