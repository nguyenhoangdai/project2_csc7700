# Neural Language Modeling with RNN, LSTM, and Transformer

This project implements and compares three sequence models—**RNN**, **LSTM**, and **Transformer**—for text generation using a prompt-completion dataset. Each model is trained separately using a SentencePiece BPE tokenizer.

---

## 📁 Files

- `rnn.py` – training and evaluation code for the RNN model  
- `lstm.py` – training and evaluation code for the LSTM model  
- `transformer.py` – training and evaluation code for the Transformer model  
- `train.jsonl` – training dataset  
- `test.jsonl` – test dataset  

---

## ✅ Requirements

- Python 3.8+
- PyTorch
- SentencePiece
- nltk
- matplotlib

Install requirements via pip:

```bash
pip3 install torch sentencepiece nltk matplotlib

```

## 🚀 How to Run

Each model can be trained and evaluated independently:

### Train and evaluate the RNN model
python3 rnn.py

### Train and evaluate the LSTM model
python3 lstm.py

### Train and evaluate the Transformer model
python3 transformer.py

## Each script:

- Loads train.jsonl and test.jsonl
- Trains the model
- Saves loss curve plots
- Prints perplexity and BLEU score
- Generates sample outputs