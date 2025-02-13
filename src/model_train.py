import argparse
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
import optuna

# --- Setup ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed_sentences = [re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F]{2}))+|@\w+|#\w+|\d+%', '', s) for s in sentences]
    processed_sentences = [re.sub(r'\d+\s*(?:year|years|month|months|day|days)\s*old|\d+-year-old|\d+\s*yo', '', s, flags=re.IGNORECASE) for s in processed_sentences]
    processed_sentences = [re.sub(r'\d+:\d+(?::\d+)?\s*(?:am|pm|AM|PM)|\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)', '', s, flags=re.IGNORECASE) for s in processed_sentences]
    processed_sentences = [re.sub(r'(?:last|next|this)\s*(?:week|month|year)', '', s, flags=re.IGNORECASE) for s in processed_sentences]
    return [word_tokenize(s.lower()) for s in processed_sentences if s.strip()]

def create_train_test_split(sentences, test_size=1000, val_size=500):
    random.shuffle(sentences)
    return sentences[test_size+val_size:], sentences[test_size:test_size+val_size], sentences[:test_size]

def build_vocabulary(sentences, min_freq=1):
    word_counts = Counter(word for sentence in sentences for word in sentence)
    vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count >= min_freq]
    return vocab, {word: idx for idx, word in enumerate(vocab)}, {idx: word for idx, word in enumerate(vocab)}

def save_vocab(vocab_data, vocab_path):
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabulary saved to: {vocab_path}")

def prepare_ngram_data(sentences, word_to_index, n):
    return [([word_to_index.get(word, word_to_index['<UNK>']) for word in sentence[i-(n-1):i]], word_to_index.get(sentence[i], word_to_index['<UNK>']))
            for sentence in sentences for i in range(n - 1, len(sentence))]

def prepare_rnn_data(sentences, word_to_index):
    return [([word_to_index.get(word, word_to_index['<UNK>']) for word in sentence[:i]], word_to_index.get(sentence[i], word_to_index['<UNK>']))
            for sentence in sentences for i in range(1, len(sentence))]

class NgramDataset(Dataset):
    def __init__(self, ngrams):
        self.ngrams = ngrams
    def __len__(self):
        return len(self.ngrams)
    def __getitem__(self, idx):
        return torch.tensor(self.ngrams[idx][0], dtype=torch.long), torch.tensor(self.ngrams[idx][1], dtype=torch.long)

class RNNSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx][0], dtype=torch.long), torch.tensor(self.sequences[idx][1], dtype=torch.long)

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
        return F.log_softmax(self.fc2(hidden), dim=1)

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
        return F.log_softmax(self.fc(output[:, -1, :]), dim=1), hidden
    def init_hidden(self, batch_size, device):
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
        return F.log_softmax(self.fc(output[:, -1, :]), dim=1), hidden
    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

def evaluate_loss(model, data_loader, criterion, model_type, device):
    model.eval()
    total_loss = 0
    word_count = 0
    with torch.no_grad():
        for batch in data_loader:
            if model_type.startswith('ffnn'):
                contexts, targets = batch
                contexts, targets = contexts.to(device), targets.to(device)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch, targets = input_seq_batch.to(device), targets.to(device)
                hidden = model.init_hidden(input_seq_batch.size(0), device)
                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            word_count += targets.size(0)
    return total_loss / word_count if word_count > 0 else 0.0

def train_model(model, train_loader, val_loader, epochs, learning_rate, model_type, model_name, device, trial_num=None):
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    model.to(device)
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    path_to_best_model = f'best_{model_name}_model.pth'

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if model_type.startswith('ffnn'):
                contexts, targets = batch
                contexts, targets = contexts.to(device), targets.to(device)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch, targets = input_seq_batch.to(device), targets.to(device)
                hidden = model.init_hidden(input_seq_batch.size(0), device)
                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % (len(train_loader) // 5) == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Training Batch Loss: {total_loss / (batch_idx + 1):.4f}')

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_val_loss = evaluate_loss(model, val_loader, criterion, model_type, device) if val_loader else 0.0
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if val_loader and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path_to_best_model)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')


    print(f"Training finished.")
    return model, best_val_loss, train_losses, val_losses


def plot_loss_curves(train_losses, val_losses, model_name, trial_num=None, final=False):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if final:
        title = f'Final Training and Validation Loss for {model_name}'
        save_path = f'{model_name}_loss_plot_final.png'
    elif trial_num is not None:
        title = f'Trial {trial_num} - Training and Validation Loss for {model_name}'
        save_path = f'{model_name}_trial_{trial_num}_loss_plot.png'
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def calculate_perplexity_per_sentence(model, sentences, word_to_index, index_to_word, criterion, model_type, device, n_gram_size=None):
    model.eval()
    sentence_perplexities = []
    total_loss = 0.0
    word_count = 0

    with torch.no_grad():
        for sentence in sentences:
            indexed_sentence = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]

            if model_type.startswith('ffnn'):
                if n_gram_size is None: raise ValueError("n_gram_size must be specified for FFNN")
                if len(indexed_sentence) < n_gram_size: continue
                sentence_ngrams = [(indexed_sentence[i - (n_gram_size - 1):i], indexed_sentence[i])
                                  for i in range(n_gram_size - 1, len(indexed_sentence))]
                if not sentence_ngrams: continue
                sentence_dataset = NgramDataset(sentence_ngrams)
                sentence_dataloader = DataLoader(sentence_dataset, batch_size=1)
                sentence_loss = 0.0
                sentence_word_count = 0
                for contexts, targets in sentence_dataloader:
                    contexts, targets = contexts.to(device), targets.to(device)
                    outputs = model(contexts)
                    loss = criterion(outputs, targets)
                    sentence_loss += loss.item() * targets.size(0)
                    sentence_word_count += targets.size(0)
                avg_sentence_loss = sentence_loss / sentence_word_count if sentence_word_count > 0 else 0

            elif model_type in ['rnn', 'lstm']:
                if len(indexed_sentence) < 2: continue
                sentence_sequences = [(indexed_sentence[:i], indexed_sentence[i]) for i in range(1, len(indexed_sentence))]
                if not sentence_sequences: continue

                sentence_dataset = RNNSequenceDataset(sentence_sequences)
                sentence_dataloader = DataLoader(sentence_dataset, batch_size=1, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])))
                sentence_loss = 0.0
                sentence_word_count = 0
                for input_seq_batch, targets in sentence_dataloader:
                    input_seq_batch, targets = input_seq_batch.to(device), targets.to(device)
                    hidden = model.init_hidden(input_seq_batch.size(0), device)
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
    # Corrected filename construction
    filepath = f"{filename}_{model_name}_{dataset_type}_perplexity.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Perplexity per sentence for {model_name} model on {dataset_type} set:\n")
        for sentence_tokens, perplexity in sentence_perplexities:
            sentence_text = " ".join(sentence_tokens)
            f.write(f"Sentence: {sentence_text}\n")
            f.write(f"Perplexity: {perplexity:.4f}\n")
            f.write("-" * 50 + "\n")
    print(f"Perplexity per sentence written to: {filepath}")


def objective(trial):
    """Objective function for Optuna optimization."""

    # Suggest hyperparameters
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 80, 128])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.7)
    num_rnn_layers = trial.suggest_int('num_rnn_layers', 1, 3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    n_gram_size = trial.suggest_categorical('n_gram_size', [3, 5])

    # Load data and prepare loaders (INSIDE the objective)
    data_text = load_data(args.data_path)
    sentences = preprocess_text(data_text)
    train_sentences, val_sentences, _ = create_train_test_split(sentences)  # No test set during tuning
    vocab, word_to_index, index_to_word = build_vocabulary(train_sentences)
    dataset_prefix = os.path.basename(args.data_path).split('.')[0].replace("-", "_").lower()

    if args.model_type.startswith('ffnn'):
        ngrams_train = prepare_ngram_data(train_sentences, word_to_index, n_gram_size)
        ngrams_val = prepare_ngram_data(val_sentences, word_to_index, n_gram_size)
        model = FFNNLM(len(vocab), embedding_dim, n_gram_size - 1, hidden_dim, dropout_prob)
        train_dataset, val_dataset = NgramDataset(ngrams_train), NgramDataset(ngrams_val)
        train_loader, val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True), DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        model_name = f'{dataset_prefix}_{args.model_type}_{n_gram_size}gram_trial_{trial.number}'

    elif args.model_type in ('rnn', 'lstm'):
        sequences_train = prepare_rnn_data(train_sentences, word_to_index)
        sequences_val = prepare_rnn_data(val_sentences, word_to_index)
        model = RNNLM(len(vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers) if args.model_type == 'rnn' else LSTMLM(len(vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers)
        train_dataset, val_dataset = RNNSequenceDataset(sequences_train), RNNSequenceDataset(sequences_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)
        model_name = f'{dataset_prefix}_{args.model_type}_trial_{trial.number}'
    else:
        raise ValueError(f"Invalid model_type: {args.model_type}")

    # Train and get losses.  Pass trial_num.
    trained_model, best_val_loss, train_losses, val_losses = train_model(model, train_loader, val_loader, args.epochs, learning_rate, args.model_type, model_name, device, trial_num=trial.number)

    # Plot loss curves *after* each trial (and before returning the value)
    plot_loss_curves(train_losses, val_losses, model_name, trial_num=trial.number)

    return best_val_loss

def main(args):
    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    print("\nOptuna Optimization Results:")
    print("  Best trial:", study.best_trial.number)
    print("  Value:", study.best_trial.value)
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"      {key}: {value}")

    # --- Retrain with Best Hyperparameters ---
    print("\nRetraining model with best hyperparameters...")
    best_params = study.best_trial.params
    
    #Correct model name
    if args.model_type.startswith('ffnn'):
      best_model_name = f"{os.path.basename(args.data_path).split('.')[0].replace('-', '_').lower()}_{args.model_type}_{best_params['n_gram_size']}gram_best"
    elif args.model_type in ('rnn', 'lstm'):
      best_model_name = f"{os.path.basename(args.data_path).split('.')[0].replace('-', '_').lower()}_{args.model_type}_best"
    else:
      raise ValueError("Invalid Model Type")

    data_text = load_data(args.data_path)
    sentences = preprocess_text(data_text)
    train_sentences, val_sentences, test_sentences = create_train_test_split(sentences) # Now using test set
    vocab, word_to_index, index_to_word = build_vocabulary(train_sentences)
    dataset_prefix = os.path.basename(args.data_path).split('.')[0].replace("-", "_").lower()
    save_vocab((vocab, word_to_index, index_to_word), f'{dataset_prefix}_vocab.pkl')

    if args.model_type.startswith('ffnn'):
        best_n_gram_size = best_params['n_gram_size']
        ngrams_train = prepare_ngram_data(train_sentences, word_to_index, best_n_gram_size)
        ngrams_val = prepare_ngram_data(val_sentences, word_to_index, best_n_gram_size)
        ngrams_test = prepare_ngram_data(test_sentences, word_to_index, best_n_gram_size)  # Prepare test data
        best_model = FFNNLM(len(vocab), best_params['embedding_dim'], best_n_gram_size - 1, best_params['hidden_dim'], best_params['dropout_prob'])
        train_dataset, val_dataset = NgramDataset(ngrams_train), NgramDataset(ngrams_val)
        train_loader, val_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, pin_memory=True), DataLoader(val_dataset, batch_size=best_params['batch_size'], pin_memory=True)
        # No test_loader during training, only for perplexity calculation
        trained_model, best_val_loss, train_losses, val_losses = train_model(best_model, train_loader, val_loader, args.epochs, best_params['learning_rate'], args.model_type, best_model_name, device)


    elif args.model_type in ('rnn', 'lstm'):
        sequences_train = prepare_rnn_data(train_sentences, word_to_index)
        sequences_val = prepare_rnn_data(val_sentences, word_to_index)
        sequences_test = prepare_rnn_data(test_sentences, word_to_index) # Prepare test data
        best_model = RNNLM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'], best_params['dropout_prob'], best_params['num_rnn_layers']) if args.model_type == 'rnn' else LSTMLM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'], best_params['dropout_prob'], best_params['num_rnn_layers'])
        train_dataset, val_dataset = RNNSequenceDataset(sequences_train), RNNSequenceDataset(sequences_val)
        train_loader, val_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True), DataLoader(val_dataset, batch_size=best_params['batch_size'], collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)
        # No test_loader during training
        trained_model, best_val_loss, train_losses, val_losses = train_model(best_model, train_loader, val_loader, args.epochs, best_params['learning_rate'], args.model_type, best_model_name, device)

    else:
        raise ValueError("Invalid Model Type")

    # Plot the final training/validation loss curve (after retraining)
    plot_loss_curves(train_losses, val_losses, best_model_name, final=True)

    # Perplexity Calculation (using the test set, after retraining)
    # and training, validation perplexity
    criterion_perplexity = nn.NLLLoss(reduction='mean', ignore_index=0)
    if args.model_type.startswith('ffnn'):
        model_type = 'ffnn'
        n_gram_size = best_params['n_gram_size'] # Use the BEST n_gram_size
        train_perplexity_sentences, avg_train_perplexity = calculate_perplexity_per_sentence(trained_model, train_sentences, word_to_index, index_to_word, criterion_perplexity, model_type, device, n_gram_size)
        val_perplexity_sentences, avg_val_perplexity = calculate_perplexity_per_sentence(trained_model, val_sentences,  word_to_index, index_to_word, criterion_perplexity, model_type,device, n_gram_size)
        test_perplexity_sentences, avg_test_perplexity = calculate_perplexity_per_sentence(trained_model, test_sentences, word_to_index, index_to_word, criterion_perplexity, model_type, device, n_gram_size)
    else:
        model_type = args.model_type
        train_perplexity_sentences, avg_train_perplexity = calculate_perplexity_per_sentence(trained_model, train_sentences, word_to_index, index_to_word, criterion_perplexity, model_type,device)
        val_perplexity_sentences, avg_val_perplexity = calculate_perplexity_per_sentence(trained_model, val_sentences,  word_to_index, index_to_word, criterion_perplexity, model_type,device)
        test_perplexity_sentences, avg_test_perplexity = calculate_perplexity_per_sentence(trained_model, test_sentences, word_to_index, index_to_word, criterion_perplexity, model_type, device)

    # Use consistent file naming (remove redundant dataset prefix) and perplexity type
    write_perplexity_to_file(dataset_prefix, best_model_name, 'train', train_perplexity_sentences)
    write_perplexity_to_file(dataset_prefix, best_model_name, 'val', val_perplexity_sentences)
    write_perplexity_to_file(dataset_prefix, best_model_name, 'test', test_perplexity_sentences)

    print(f"Model {args.model_type} - Avg Train Perplexity: {avg_train_perplexity:.4f}")
    print(f"Model {args.model_type} - Avg Val Perplexity: {avg_val_perplexity:.4f}")
    print(f"Model {args.model_type} - Avg Test Perplexity: {avg_test_perplexity:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a language model with Optuna.')
    parser.add_argument('--model_type', type=str, required=True, choices=['ffnn_3gram', 'ffnn_5gram', 'rnn', 'lstm'], help='Type of model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data.')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of Optuna trials.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs.')  # Now required
    args = parser.parse_args()

    # Automatic device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    main(args)