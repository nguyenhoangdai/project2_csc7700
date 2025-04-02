import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sentencepiece as spm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu

class TextGenerationPipeline:
    """
    Single-class pipeline that:
      - Trains a SentencePiece BPE tokenizer
      - Loads data from JSONL
      - Provides RNN / LSTM / Transformer models
      - Trains with:
          * CrossEntropyLoss
          * AdamW
          * Batch size = 128
          * 30 epochs (with early stopping + LR scheduler)
      - Evaluates with perplexity (PPL) and BLEU
    """

    def __init__(
        self,
        train_file: str = "train.jsonl",
        test_file: str = "test.jsonl",
        model_prefix: str = "spm_model",
        vocab_size: int = 10000,
        max_seq_length: int = 128,
        batch_size: int = 128,   # Start at 128 as recommended
        embed_dim: int = 256,
        hidden_dim: int = 512, # could be adjusted
        num_layers: int = 1,
        num_heads: int = 4,
        transformer_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 30,       # 30 epochs
        patience: int = 3,      # for early stopping
        device: str = None,
    ):
        self.train_file = train_file
        self.test_file = test_file
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Will be set once the tokenizer is trained/loaded:
        self.sp_model = None

        # We'll store losses here if you want to plot them
        self.train_losses = []
        self.val_losses = []

    # ----------------------------------------------------------------
    # 1. Tokenizer
    # ----------------------------------------------------------------
    def train_tokenizer(self):
        """
        Trains a SentencePiece BPE tokenizer on 'prompt' + 'completion' from train_file.
        """
        temp_txt_file = "train_text_for_tokenizer.txt"
        with open(self.train_file, 'r', encoding='utf-8') as f_in, open(temp_txt_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                data = json.loads(line)
                prompt_text = data['prompt'].strip()
                completion_text = data['completion'].strip()
                combined_line = f"{prompt_text} {completion_text}"
                f_out.write(combined_line + "\n")

        spm.SentencePieceTrainer.Train(
            input=temp_txt_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            bos_id=-1,   # no explicit BOS
            eos_id=1     # EOS token = 1
        )

        os.remove(temp_txt_file)

        sp = spm.SentencePieceProcessor()
        sp.Load(f"{self.model_prefix}.model")
        self.sp_model = sp

    # ----------------------------------------------------------------
    # 2. Dataset
    # ----------------------------------------------------------------
    class PromptCompletionDataset(Dataset):
        """
        Reads lines of form:
          {"prompt": "...", "completion": "..."}
        Concatenates them, encodes with SentencePiece, returns the token IDs.
        """
        def __init__(self, file_path, sp_model, max_length=128):
            super().__init__()
            self.samples = []
            self.sp_model = sp_model
            self.max_length = max_length

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    prompt_text = item['prompt'].strip()
                    completion_text = item['completion'].strip()
                    combined = f"{prompt_text} {completion_text}"
                    token_ids = sp_model.EncodeAsIds(combined)
                    token_ids = token_ids[:max_length]
                    self.samples.append(torch.tensor(token_ids, dtype=torch.long))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    @staticmethod
    def collate_fn(batch, pad_id=0):
        """
        Collate function to pad the batch to the same length.
        """
        max_len = max(len(seq) for seq in batch)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(batch):
            padded[i, :len(seq)] = seq
        return padded

    def load_datasets(self):
        """
        Returns train DataLoader, test DataLoader, plus the raw train & test Dataset objects.
        """
        train_ds = self.PromptCompletionDataset(
            self.train_file, self.sp_model, max_length=self.max_seq_length
        )
        test_ds = self.PromptCompletionDataset(
            self.test_file, self.sp_model, max_length=self.max_seq_length
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: self.collate_fn(x, pad_id=0)
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: self.collate_fn(x, pad_id=0)
        )
        return train_loader, test_loader, train_ds, test_ds

    # ----------------------------------------------------------------
    # 3. Model Definitions
    # ----------------------------------------------------------------
    class BaseRNNModel(nn.Module):
        """
        Shared class for RNN / LSTM with embedding + RNN + FC
        """
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, rnn_type='RNN'):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            if rnn_type.lower() == 'rnn':
                self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
            else:
                self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)
            self.rnn_type = rnn_type

        def forward(self, x, hidden=None):
            embeds = self.embedding(x)
            out, hidden = self.rnn(embeds, hidden)
            logits = self.fc(out)
            return logits, hidden

        def generate(self, text, sp_model, device, max_length=50, temperature=1.0):
            self.eval()
            tokens = sp_model.EncodeAsIds(text)
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            hidden = None
            generated_ids = tokens[:]

            for _ in range(max_length):
                logits, hidden = self.forward(input_tensor, hidden)
                # only last timestep
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated_ids.append(next_token)

                if next_token == sp_model.eos_id():
                    break

                input_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
            return sp_model.DecodeIds(generated_ids)

    class TransformerModel(nn.Module):
        """
        Simple Transformer encoder-based language model.
        """
        def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.embed_dim = embed_dim
            self.pos_enc = self.PositionalEncoding(embed_dim, max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(embed_dim, vocab_size)
            self.max_len = max_len

        def forward(self, x):
            embeds = self.embedding(x) * math.sqrt(self.embed_dim)
            embeds = self.pos_enc(embeds)
            # input shape for transformer: (seq_len, batch, embed_dim)
            embeds = embeds.transpose(0, 1)
            out = self.transformer(embeds)
            out = out.transpose(0, 1)
            logits = self.fc(out)
            return logits

        def generate(self, text, sp_model, device, max_length=50, temperature=1.0):
            self.eval()
            tokens = sp_model.EncodeAsIds(text)
            generated_ids = tokens[:]

            for _ in range(max_length):
                context = generated_ids[-self.max_len:]
                input_tensor = torch.tensor([context], dtype=torch.long).to(device)
                logits = self.forward(input_tensor)
                # get last timestep
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated_ids.append(next_token)
                if next_token == sp_model.eos_id():
                    break

            return sp_model.DecodeIds(generated_ids)

        class PositionalEncoding(nn.Module):
            """
            Sinusoidal Positional Encoding
            """
            def __init__(self, d_model, max_len=512):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
                self.register_buffer('pe', pe)

            def forward(self, x):
                seq_len = x.size(1)
                x = x + self.pe[:, :seq_len]
                return x

    def build_model(self, model_type):
        """
        model_type in ['rnn', 'lstm', 'transformer']
        """
        vocab_size = self.sp_model.GetPieceSize()
        if model_type.lower() == 'rnn':
            return self.BaseRNNModel(
                vocab_size, self.embed_dim, self.hidden_dim, self.num_layers, rnn_type='rnn'
            )
        elif model_type.lower() == 'lstm':
            return self.BaseRNNModel(
                vocab_size, self.embed_dim, self.hidden_dim, self.num_layers, rnn_type='lstm'
            )
        elif model_type.lower() == 'transformer':
            return self.TransformerModel(
                vocab_size,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                num_layers=self.transformer_layers,
                max_len=self.max_seq_length
            )
        else:
            raise ValueError("model_type must be one of ['rnn','lstm','transformer'].")

    # ----------------------------------------------------------------
    # 4. Training (30 epochs, Early Stopping, LR Scheduler)
    # ----------------------------------------------------------------
    def train_model(self, model, train_loader, val_loader):
        """
        Trains the model with:
          - CrossEntropyLoss
          - AdamW
          - 30 epochs (self.epochs)
          - LR Scheduler
          - Early Stopping
        """
        model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # LR scheduler on plateau
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

        best_val_loss = float('inf')
        no_improve_count = 0
        self.train_losses = []
        self.val_losses = []

        for epoch in range(1, self.epochs + 1):
            # ---------------------------
            # Training
            # ---------------------------
            model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                optimizer.zero_grad()
                if hasattr(model, 'rnn'):  # RNN or LSTM
                    outputs, _ = model(inputs)
                else:  # Transformer
                    outputs = model(inputs)

                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # ---------------------------
            # Validation
            # ---------------------------
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(self.device)
                    val_inputs = val_batch[:, :-1]
                    val_targets = val_batch[:, 1:]
                    if hasattr(model, 'rnn'):
                        val_outputs, _ = model(val_inputs)
                    else:
                        val_outputs = model(val_inputs)
                    val_loss = criterion(
                        val_outputs.reshape(-1, val_outputs.shape[-1]),
                        val_targets.reshape(-1)
                    )
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)

            # Step scheduler
            scheduler.step(avg_val_loss)

            # Print epoch summary
            print(f"Epoch [{epoch}/{self.epochs}] "
                  f"- Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    print("Early stopping triggered!")
                    break

    # ----------------------------------------------------------------
    # 5. Evaluation: Perplexity & BLEU
    # ----------------------------------------------------------------
    def evaluate_perplexity(self, model, data_loader):
        """
        Compute average test loss, then compute PPL = exp(loss).
        """
        model.to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        total_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                if hasattr(model, 'rnn'):
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        ppl = np.exp(avg_loss)
        return avg_loss, ppl

    def compute_bleu(self, model, dataset, num_samples=20):
        """
        Compute BLEU by generating from half the sample as a 'prompt'
        and comparing to the entire sample as reference.
        """
        model.eval()
        scores = []

        # We'll just check the first num_samples in the dataset
        subset = dataset[:num_samples]
        for sample in subset:
            # Reference is the entire line
            ref_text = self.sp_model.DecodeIds(sample.tolist())
            ref_tokens = [ref_text.split()]  # BLEU expects list of references

            # Use half the sequence as the prompt
            half_len = max(1, len(sample) // 2)
            prompt_ids = sample[:half_len].tolist()
            prompt_text = self.sp_model.DecodeIds(prompt_ids)

            generated_text = model.generate(
                text=prompt_text,
                sp_model=self.sp_model,
                device=self.device,
                max_length=50,
                temperature=1.0
            )
            candidate_tokens = generated_text.split()

            score = sentence_bleu(ref_tokens, candidate_tokens)
            scores.append(score)

        return float(np.mean(scores))

    # ----------------------------------------------------------------
    # 6. Utility: Plot Loss Curves
    # ----------------------------------------------------------------
    def plot_loss_curves(self):
        """
        Plots training/validation loss over epochs.
        """
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("lstm.png", dpi=300, bbox_inches='tight')

        # plt.show()

    # ----------------------------------------------------------------
    # 7. Demonstration
    # ----------------------------------------------------------------
    def run_example(self):
        """
        Demonstrates the entire flow:
          1. Train tokenizer
          2. Load data
          3. Build a model (RNN, LSTM, Transformer)
          4. Train for 30 epochs (with early stopping & LR scheduler)
          5. Evaluate perplexity
          6. Evaluate BLEU
          7. Plot curves
          8. Generate sample text
        """
        # 1) Train (or load) tokenizer
        self.train_tokenizer()

        # 2) Load data
        train_loader, test_loader, train_ds, test_ds = self.load_datasets()

        # 3) Choose a model type: 'rnn', 'lstm', or 'transformer'
        model = self.build_model(model_type='lstm')

        # 4) Train
        self.train_model(model, train_loader, test_loader)

        # 5) Evaluate perplexity on test
        test_loss, test_ppl = self.evaluate_perplexity(model, test_loader)
        print(f"[Test Perplexity] Loss: {test_loss:.4f} | PPL: {test_ppl:.2f}")

        # 6) BLEU
        bleu = self.compute_bleu(model, test_ds, num_samples=20)
        print(f"[BLEU] Score on 20 samples: {bleu:.4f}")

        # 7) Plot
        self.plot_loss_curves()

        # 8) Generate text from the required prompt
        sample_prompt = "Which do you prefer? Dogs or cats?"
        result = model.generate(
            text=sample_prompt,
            sp_model=self.sp_model,
            device=self.device,
            max_length=50,
            temperature=1.0
        )
        print(f"Required Prompt ({sample_prompt}) ->\n{result}\n")

        # Generate text from a custom prompt
        custom_prompt = "Once upon a time, in a quiet village,"
        custom_result = model.generate(
            text=custom_prompt,
            sp_model=self.sp_model,
            device=self.device,
            max_length=50,
            temperature=1.0
        )
        print(f"Custom Prompt ({custom_prompt}) ->\n{custom_result}\n")

# ---------------------
# ---------------------
if __name__ == "__main__":
    pipeline = TextGenerationPipeline(
        train_file="train.jsonl",
        test_file="test.jsonl"
    )
    pipeline.run_example()
