# model_train.py
import argparse  # Import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import numpy as np
import random
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import os
import pickle

# --- Setup and Data Loading ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data(file_path):
    """Loads text data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def preprocess_text(text):
    """Tokenizes text into sentences and words, lowercases words and removing special characters."""
    sentences = sent_tokenize(text)
    processed_sentences = []
    for s in sentences:
        # Remove URLs
        s = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F]{2}))+', '', s)
        # Remove mentions
        s = re.sub(r'@\w+', '', s)
        # Remove hashtags
        s = re.sub(r'#\w+', '', s)
        # Remove percentages
        s = re.sub(r'\d+%', '', s)
        # Remove age values
        s = re.sub(r'\d+\s*(?:year|years|month|months|day|days)\s*old', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\d+-year-old', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\d+\s*yo', '', s, flags=re.IGNORECASE)
        # Remove time expressions
        s = re.sub(r'\d+:\d+(?::\d+)?\s*(?:am|pm|AM|PM)', '', s)
        s = re.sub(r'\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)', '', s, flags=re.IGNORECASE)
        s = re.sub(r'(?:last|next|this)\s*(?:week|month|year)', '', s, flags=re.IGNORECASE)

        processed_sentences.append(s)

    tokenized_sentences = [word_tokenize(s.lower()) for s in processed_sentences if s.strip()]
    return tokenized_sentences

def create_train_test_split(sentences, test_size=1000, val_size=500):
    """Splits sentences into training, validation, and test sets."""
    random.shuffle(sentences)
    test_sentences = sentences[:test_size]
    val_sentences = sentences[test_size:test_size+val_size]
    train_sentences = sentences[test_size+val_size:]
    return train_sentences, val_sentences, test_sentences

def build_vocabulary(sentences, min_freq=1):
    """Builds vocabulary from sentences."""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)

    vocab = ['<PAD>', '<UNK>']
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab.append(word)

    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word_to_index, index_to_word

def save_vocab(vocab_data, vocab_path):
    """Saves vocabulary data to a file."""
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabulary saved to: {vocab_path}")

# --- Data Preparation for Models ---
def prepare_ngram_data(sentences, word_to_index, n):
    """Prepares n-gram data from sentences."""
    ngrams = []
    for sentence in sentences:
        indexed_sentence = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
        if len(indexed_sentence) < n:
            continue
        for i in range(n - 1, len(indexed_sentence)):
            context = indexed_sentence[i - (n - 1):i]
            target = indexed_sentence[i]
            ngrams.append((context, target))
    return ngrams

def prepare_rnn_data(sentences, word_to_index):
    """Prepares RNN sequence data from sentences."""
    sequences = []
    for sentence in sentences:
        indexed_sentence = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
        for i in range(1, len(indexed_sentence)):
            input_seq = indexed_sentence[:i]
            target_word = indexed_sentence[i]
            sequences.append((input_seq, target_word))
    return sequences

# --- Dataset Classes ---
class NgramDataset(Dataset):
    def __init__(self, ngrams):
        self.ngrams = ngrams

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        context, target = self.ngrams[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class RNNSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_word = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_word, dtype=torch.long)

# --- Model Definitions ---
class FFNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, dropout_prob=0.2):
        super(FFNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.fc1_norm = nn.LayerNorm(hidden_dim)
        self.dropout_hidden = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context):
        emb = self.dropout_emb(self.embedding_norm(self.embedding(context))).view(context.size(0), -1)
        hidden = F.relu(self.fc1_norm(self.fc1(emb)))
        hidden = self.dropout_hidden(hidden)
        output = F.log_softmax(self.fc2(hidden), dim=1)
        return output

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.rnn_output_norm = nn.LayerNorm(hidden_dim)
        self.dropout_rnn_output = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_seq, hidden):
        emb = self.dropout_emb(self.embedding_norm(self.embedding(input_seq)))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout_rnn_output(self.rnn_output_norm(output))
        output_last_step = output[:, -1, :]
        output_logprobs = F.log_softmax(self.fc(output_last_step), dim=1)
        return output_logprobs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2):
        super(LSTMLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.lstm_output_norm = nn.LayerNorm(hidden_dim)
        self.dropout_lstm_output = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_seq, hidden):
        emb = self.dropout_emb(self.embedding_norm(self.embedding(input_seq)))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout_lstm_output(self.lstm_output_norm(output))
        output_last_step = output[:, -1, :]
        output_logprobs = F.log_softmax(self.fc(output_last_step), dim=1)
        return output_logprobs, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

# --- Training and Evaluation Functions ---
def evaluate_loss(model, data_loader, criterion, model_type, device):
    """Evaluates the model."""
    model.eval()
    total_loss = 0
    word_count = 0
    if data_loader is None:
        return 0.0
    with torch.no_grad():
        for batch in data_loader:
            if model_type.startswith('ffnn'):  # Check for ffnn variants
                contexts, targets = batch
                contexts, targets = contexts.to(device), targets.to(device)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch, targets = input_seq_batch.to(device), targets.to(device)
                hidden = model.init_hidden(input_seq_batch.size(0))
                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            word_count += targets.size(0)
    avg_loss = total_loss / word_count if word_count > 0 else 0
    return avg_loss

def train_model(model, train_loader, val_loader, epochs, learning_rate, model_type, model_name):
    """Trains FFNN, RNN, and LSTM models."""
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    model.to(device)
    model.train()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    path_to_best_model = f'best_{model_name}_model.pth'
    epoch_batches_print = len(train_loader) // 5 if len(train_loader) > 5 else 1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if model_type.startswith('ffnn'): # Check for ffnn variants
                contexts, targets = batch
                contexts, targets = contexts.to(device), targets.to(device)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch, targets = input_seq_batch.to(device), targets.to(device)
                hidden = model.init_hidden(input_seq_batch.size(0))
                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % epoch_batches_print == 0:
                avg_loss_batch = total_loss / (batch_idx + 1)
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Training Batch Loss: {avg_loss_batch:.4f}')

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = 0
        if val_loader is not None:
            avg_val_loss = evaluate_loss(model, val_loader, criterion, model_type, device)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), path_to_best_model)

        if val_loader is not None:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        else:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f"Training finished.")
    if val_loader is not None:
        print(f"Best Validation Loss: {best_val_loss:.4f}.  Loading best model from {path_to_best_model}")
        model.load_state_dict(torch.load(path_to_best_model))

    if val_loader is not None:
        plot_loss_curves(train_losses, val_losses, model_name)
    else:
        plot_loss_curves(train_losses, [], model_name)
    return model

def plot_loss_curves(train_losses, val_losses, model_name):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.legend()
    plt.show()

def calculate_perplexity_per_sentence(model, sentences, word_to_index, index_to_word, criterion, model_type, n_gram_size=None):
    """Calculates perplexity for each sentence."""
    model.eval()
    sentence_perplexities = []
    total_loss = 0.0
    word_count = 0

    with torch.no_grad():
        for sentence in sentences:
            indexed_sentence = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]

            if model_type.startswith('ffnn'):
                if n_gram_size is None:
                    raise ValueError("n_gram_size must be specified for FFNN model")
                if len(indexed_sentence) < n_gram_size:
                    continue

                sentence_ngrams = []
                for i in range(n_gram_size - 1, len(indexed_sentence)):
                    context = indexed_sentence[i - (n_gram_size - 1):i]
                    target = indexed_sentence[i]
                    sentence_ngrams.append((context, target))

                if not sentence_ngrams:
                    continue

                sentence_dataset = NgramDataset(sentence_ngrams)
                sentence_dataloader = DataLoader(sentence_dataset, batch_size=1)

                sentence_loss = 0.0
                sentence_word_count = 0
                for batch in sentence_dataloader:
                    contexts, targets = batch
                    contexts, targets = contexts.to(device), targets.to(device)
                    outputs = model(contexts)
                    loss = criterion(outputs, targets)
                    sentence_loss += loss.item() * targets.size(0)
                    sentence_word_count += targets.size(0)
                avg_sentence_loss = sentence_loss / sentence_word_count if sentence_word_count > 0 else 0

            elif model_type in ['rnn', 'lstm']:
                if len(indexed_sentence) < 2:
                    continue

                sentence_sequences = []
                for i in range(1, len(indexed_sentence)):
                    input_seq = indexed_sentence[:i]
                    target_word = indexed_sentence[i]
                    sentence_sequences.append((input_seq, target_word))

                if not sentence_sequences:
                    continue
                sentence_dataset = RNNSequenceDataset(sentence_sequences)
                sentence_dataloader = DataLoader(sentence_dataset, batch_size=1, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])))

                sentence_loss = 0.0
                sentence_word_count = 0
                for batch in sentence_dataloader:
                    input_seq_batch, targets = batch
                    input_seq_batch, targets = input_seq_batch.to(device), targets.to(device)
                    hidden = model.init_hidden(input_seq_batch.size(0))
                    outputs, _ = model(input_seq_batch, hidden)
                    loss = criterion(outputs, targets)
                    sentence_loss += loss.item() * targets.size(0)
                    sentence_word_count += targets.size(0)
                avg_sentence_loss = sentence_loss / sentence_word_count if sentence_word_count > 0 else 0

            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            sentence_perplexity = torch.exp(torch.tensor(avg_sentence_loss)).item() if avg_sentence_loss > 0 else float('inf')
            sentence_perplexities.append((sentence, sentence_perplexity))
            total_loss += sentence_loss
            word_count += sentence_word_count

    avg_perplexity = torch.exp(torch.tensor(total_loss / word_count)).item() if word_count > 0 else float('inf')
    return sentence_perplexities, avg_perplexity

def write_perplexity_to_file(filename, model_name, dataset_type, sentence_perplexities):
    """Writes sentence perplexities to a file."""
    filepath = f"{filename}_{model_name}_{dataset_type}_perplexity.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Perplexity per sentence for {model_name} model on {dataset_type} set:\n")
        for sentence_tokens, perplexity in sentence_perplexities:
            sentence_text = " ".join(sentence_tokens)
            f.write(f"Sentence: {sentence_text}\n")
            f.write(f"Perplexity: {perplexity:.4f}\n")
            f.write("-" * 50 + "\n")
    print(f"Perplexity per sentence written to: {filepath}")

def main(args):
    """Main function to handle training based on command-line arguments."""

    # --- Data Loading and Preprocessing ---
    data_text = load_data(args.data_path)
    sentences = preprocess_text(data_text)
    train_sentences, val_sentences, test_sentences = create_train_test_split(sentences)
    vocab, word_to_index, index_to_word = build_vocabulary(train_sentences)

    # Determine dataset prefix from the data path (for saving files)
    dataset_prefix = os.path.basename(args.data_path).split('.')[0]  # e.g., "Pride-and-Prejudice-Jane-Austen"
    dataset_prefix = dataset_prefix.replace("-", "_").lower() #for valid file name

    # Save Vocabulary
    vocab_data = (vocab, word_to_index, index_to_word)
    save_vocab(vocab_data, f'{dataset_prefix}_vocab.pkl')

    # --- Hyperparameters (can be made into argparse arguments if needed) ---
    embedding_dim = 80
    hidden_dim = 256
    learning_rate = 0.001
    batch_size = 32
    dropout_prob = 0.5
    num_rnn_layers = 2
    n_gram_size_3 = 3
    n_gram_size_5 = 5

    # --- Model Training based on args.model_type ---
    if args.model_type == 'ffnn_3gram':
        ngrams_train = prepare_ngram_data(train_sentences, word_to_index, n_gram_size_3)
        ngrams_val = prepare_ngram_data(val_sentences, word_to_index, n_gram_size_3)
        model = FFNNLM(len(vocab), embedding_dim, n_gram_size_3 - 1, hidden_dim, dropout_prob)
        train_dataset = NgramDataset(ngrams_train)
        val_dataset = NgramDataset(ngrams_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        trained_model = train_model(model, train_loader, val_loader, args.epochs, learning_rate, 'ffnn', f'{dataset_prefix}_ffnn_3gram')

    elif args.model_type == 'ffnn_5gram':
        ngrams_train = prepare_ngram_data(train_sentences, word_to_index, n_gram_size_5)
        ngrams_val = prepare_ngram_data(val_sentences, word_to_index, n_gram_size_5)
        model = FFNNLM(len(vocab), embedding_dim, n_gram_size_5 - 1, hidden_dim, dropout_prob)
        train_dataset = NgramDataset(ngrams_train)
        val_dataset = NgramDataset(ngrams_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        trained_model = train_model(model, train_loader, val_loader, args.epochs, learning_rate, 'ffnn', f'{dataset_prefix}_ffnn_5gram')

    elif args.model_type == 'rnn':
        rnn_sequences_train = prepare_rnn_data(train_sentences, word_to_index)
        rnn_sequences_val = prepare_rnn_data(val_sentences, word_to_index)
        model = RNNLM(len(vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers)
        train_dataset = RNNSequenceDataset(rnn_sequences_train)
        val_dataset = RNNSequenceDataset(rnn_sequences_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)
        trained_model = train_model(model, train_loader, val_loader, args.epochs, learning_rate, 'rnn', f'{dataset_prefix}_rnn')

    elif args.model_type == 'lstm':
        rnn_sequences_train = prepare_rnn_data(train_sentences, word_to_index)
        rnn_sequences_val = prepare_rnn_data(val_sentences, word_to_index)
        model = LSTMLM(len(vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers)
        train_dataset = RNNSequenceDataset(rnn_sequences_train)
        val_dataset = RNNSequenceDataset(rnn_sequences_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)
        trained_model = train_model(model, train_loader, val_loader, args.epochs, learning_rate, 'lstm', f'{dataset_prefix}_lstm')

    else:
        raise ValueError(f"Invalid model_type: {args.model_type}")

    #--- Perplexity Calculation ---
    criterion_perplexity = nn.NLLLoss(reduction='mean', ignore_index=0)
    if args.model_type.startswith('ffnn'):
      model_type = 'ffnn'
      n_gram_size = int(args.model_type.split('_')[-1].replace('gram', ''))
      train_perplexity_sentences, avg_train_perplexity = calculate_perplexity_per_sentence(trained_model, train_sentences, word_to_index, index_to_word, criterion_perplexity, model_type, n_gram_size)
      val_perplexity_sentences, avg_val_perplexity = calculate_perplexity_per_sentence(trained_model, val_sentences,  word_to_index, index_to_word, criterion_perplexity, model_type, n_gram_size)
      test_perplexity_sentences, avg_test_perplexity = calculate_perplexity_per_sentence(trained_model, test_sentences,  word_to_index, index_to_word, criterion_perplexity, model_type, n_gram_size)

    else:
      model_type = args.model_type
      train_perplexity_sentences, avg_train_perplexity = calculate_perplexity_per_sentence(trained_model, train_sentences, word_to_index, index_to_word, criterion_perplexity, model_type=model_type)
      val_perplexity_sentences, avg_val_perplexity = calculate_perplexity_per_sentence(trained_model, val_sentences,  word_to_index, index_to_word, criterion_perplexity, model_type=model_type)
      test_perplexity_sentences, avg_test_perplexity = calculate_perplexity_per_sentence(trained_model, test_sentences,  word_to_index, index_to_word, criterion_perplexity, model_type=model_type)

    write_perplexity_to_file(dataset_prefix, args.model_type, 'train', train_perplexity_sentences)
    write_perplexity_to_file(dataset_prefix, args.model_type, 'val', val_perplexity_sentences)
    write_perplexity_to_file(dataset_prefix, args.model_type, 'test', test_perplexity_sentences)
    print(f"Model {args.model_type} - Avg Train Perplexity: {avg_train_perplexity:.4f}, Avg Val Perplexity: {avg_val_perplexity:.4f}, Avg Test Perplexity: {avg_test_perplexity:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a language model.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['ffnn_3gram', 'ffnn_5gram', 'rnn', 'lstm'],
                        help='Type of model to train (ffnn_3gram, ffnn_5gram, rnn, lstm).')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data file.')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of training epochs.')
    # You could add more argparse arguments for hyperparameters like learning_rate, etc.
    args = parser.parse_args()
    main(args)