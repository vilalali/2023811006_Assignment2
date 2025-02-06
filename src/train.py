import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import numpy as np
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import os
import argparse
import optuna  # Import Optuna for hyperparameter tuning
import csv
from tokenizer import Tokenizer #Import the tokenizer

# --- 1. Setup and Data Loading ---
SEED = 44
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
print(f"Using device: {device}, Number of GPUs available: {n_gpus}")
multi_gpu = n_gpus > 1
tokenizer = Tokenizer() # Instantiate the tokenizer

def load_data(file_path):
    """Loads text data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def preprocess_text(text):
    """Tokenizes text into sentences and words, lowercasing words."""
    sentences = tokenizer.tokenize_content(text, remove_puctuations = True)
    return sentences

def create_train_test_split(sentences, test_size=1000, seed = SEED):
    """Splits sentences into training, validation, and test sets."""
    random.seed(seed)
    random.shuffle(sentences)
    test_sentences = sentences[:test_size]
    train_sentences = sentences[test_size:]
    return train_sentences, test_sentences

def build_vocabulary(sentences, min_freq=1):
    """Builds vocabulary from sentences."""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    vocab = ['<PAD>', '<UNK>'] # PAD token at index 0
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab.append(word)

    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word_to_index, index_to_word

# --- 2. Data Preparation for Models ---

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

# --- 3. Dataset Classes ---

class NgramDataset(Dataset):
    def __init__(self, ngrams):
        self.ngrams = ngrams

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        context, target = self.ngrams[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long) # Corrected: Return tensors

class RNNSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_word = self.sequences[idx]
        return input_seq, target_word # Return raw lists/ints, tensors created in DataLoader


# --- 4. Model Definitions ---

class FFNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, dropout_prob):
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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2): # num_layers=2
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout_prob) # dropout in RNN layer
        self.rnn_output_norm = nn.LayerNorm(hidden_dim)
        self.dropout_rnn_output = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers # Store num_layers

    def forward(self, input_seq, hidden):
        emb = self.dropout_emb(self.embedding_norm(self.embedding(input_seq)))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout_rnn_output(self.rnn_output_norm(output))
        output_last_step = output[:, -1, :]
        output_logprobs = F.log_softmax(self.fc(output_last_step), dim=1)
        return output_logprobs, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device) # Initialize hidden for num_layers

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2): # num_layers=2
        super(LSTMLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout_prob) # dropout in LSTM layer
        self.lstm_output_norm = nn.LayerNorm(hidden_dim)
        self.dropout_lstm_output = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers # Store num_layers


    def forward(self, input_seq, hidden):
        emb = self.dropout_emb(self.embedding_norm(self.embedding(input_seq)))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout_lstm_output(self.lstm_output_norm(output))
        output_last_step = output[:, -1, :]
        output_logprobs = F.log_softmax(self.fc(output_last_step), dim=1)
        return output_logprobs, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device), # Initialize hidden for num_layers
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)) # Initialize cell state for num_layers

# --- 5. Training and Evaluation Functions ---

def collate_fn_rnn(batch):
    """Pads sequences within a batch and stacks targets."""
    input_seqs, targets = zip(*batch)
    padded_input_seqs = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in input_seqs], batch_first=True, padding_value=0)
    targets = torch.stack([torch.tensor(target, dtype=torch.long) for target in targets])
    return padded_input_seqs, targets
def collate_fn_rnn_loss(batch):
    """Pads sequences within a batch and stacks targets."""
    input_seqs, targets = zip(*batch)
    padded_input_seqs = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in input_seqs], batch_first=True, padding_value=0)
    targets = torch.stack([torch.tensor(target, dtype=torch.long) for target in targets])
    return padded_input_seqs, targets

def evaluate_loss(model, data_loader, criterion, model_type, device, multi_gpu=False):
    """Evaluates the model on the given data loader and returns the average loss."""
    model.eval()
    total_loss = 0
    word_count = 0
    with torch.no_grad():
        for batch in data_loader:
            if model_type == 'ffnn':
                contexts, targets = batch
                contexts = contexts.to(device, non_blocking=True) # non_blocking=True for async GPU transfer
                targets = targets.to(device, non_blocking=True)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch = input_seq_batch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if multi_gpu:
                    hidden = model.module.init_hidden(input_seq_batch.size(0), device)
                else:
                    hidden = model.init_hidden(input_seq_batch.size(0), device)
                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            if multi_gpu:
                loss = loss.mean() # Average loss over GPUs if using DataParallel
            total_loss += loss.item() * targets.size(0)
            word_count += targets.size(0)
    avg_loss = total_loss / word_count if word_count > 0 else 0
    return avg_loss
def evaluate_loss_per_sentence(model, data_loader, criterion, model_type, device, multi_gpu=False):
    """Evaluates the model on the given data loader and returns sentence-wise loss and perplexity."""
    model.eval()
    all_losses = []
    all_perplexities = []
    with torch.no_grad():
         for batch in data_loader:
            if model_type == 'ffnn':
                contexts, targets = batch
                contexts = contexts.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch = input_seq_batch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if multi_gpu:
                    hidden = model.module.init_hidden(input_seq_batch.size(0), device)
                else:
                   hidden = model.init_hidden(input_seq_batch.size(0), device)
                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            
            if multi_gpu:
                losses = criterion(outputs, targets).mean(dim=0).cpu().numpy() # Average loss over batch dimension for multi-GPU
            else:
                losses = criterion(outputs, targets).cpu().numpy()
            
            # Ensure losses is always a 1D array
            if losses.ndim == 0:  # handle cases if loss is a single scalar
               losses = np.array([losses])
            perplexities = np.exp(losses)
            all_losses.extend(losses)
            all_perplexities.extend(perplexities)
    return all_losses, all_perplexities

# Separate save_model function (to be placed outside of train_model):
def save_model(model, model_path, vocab, word_to_index, model_type, best_params, n_gram=None):
    state = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'word_to_index': word_to_index,
        'model_type': model_type,
        'best_params' : best_params,
        'n_gram': n_gram
    }
    torch.save(state, model_path)

"""Generic training function for FFNN, RNN, and LSTM models with early stopping."""
def train_model(model, train_loader, val_loader, epochs, learning_rate, model_type, model_name, patience=7, device=None, multi_gpu=False, vocab=None, word_to_index=None, best_params = None, n_gram = None, args = None, train_sentences = None, test_sentences = None):
    criterion = nn.NLLLoss(ignore_index=0, reduction = 'mean')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    model.to(device)
    if multi_gpu:
        model = nn.DataParallel(model) # Wrap model for multi-GPU

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    path_to_best_model = os.path.join(args.model_dir, f'best_{model_name}_model.pth')
    epoch_batches_print = len(train_loader) // 5 if len(train_loader) > 5 else 1
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train() # Ensure train mode is set at the beginning of each epoch
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if model_type == 'ffnn':
                contexts, targets = batch
                contexts = contexts.to(device, non_blocking=True) # non_blocking=True
                targets = targets.to(device, non_blocking=True)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch = input_seq_batch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Correct hidden state initialization for DataParallel
                if multi_gpu:
                    # Get sub-batch size for current GPU (replica)
                    sub_batch_size = input_seq_batch.size(0) // (torch.cuda.device_count() if torch.cuda.is_available() else 1 ) # Get GPU count for multi-gpu
                    # Initialize hidden state with sub-batch size
                    hidden = model.module.init_hidden(sub_batch_size, device)
                else:
                    hidden = model.init_hidden(input_seq_batch.size(0), device)

                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            if multi_gpu:
                loss = loss.mean() # Average loss over GPUs if using DataParallel
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % epoch_batches_print == 0:
                 avg_loss_batch = total_loss / (batch_idx+1)
                 print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Training Batch Loss: {avg_loss_batch:.4f}')

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = evaluate_loss(model, val_loader, criterion, model_type, device, multi_gpu=multi_gpu)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if multi_gpu:
                torch.save(model.module.state_dict(), path_to_best_model) # Save state_dict of the module
            else:
                torch.save(model.state_dict(), path_to_best_model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
    
    # Store the train and val losses in the model class
    model.train_losses = train_losses
    model.val_losses = val_losses
    
    # --- Perplexity Calculation and Reporting ---
    
    # Calculate and save perplexity for the training set
    train_losses, train_perplexities = evaluate_loss_per_sentence(model, train_loader, nn.NLLLoss(ignore_index=0, reduction = 'none'), model_type, device, multi_gpu=multi_gpu)
    avg_train_perplexity = np.mean(train_perplexities)

    train_output_dir = os.path.join(args.output_dir, "train_perplexity")
    os.makedirs(train_output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    train_output_path = os.path.join(train_output_dir, f"{args.student_id}_{model_name}_Perplexity_Train_{os.path.basename(args.corpus_path).split('.')[0]}.csv")
    
    # Save the training sentences to a file
    train_sentences_path = os.path.join(train_output_dir, f"{args.student_id}_train_sentences_{os.path.basename(args.corpus_path).split('.')[0]}.txt")
    with open(train_sentences_path, 'w', encoding = 'utf-8') as f:
       for sent in train_sentences:
           f.write(f"{' '.join(sent)}\n") # Join tokens
    print(f"Training sentences saved to: {train_sentences_path}")

    with open(train_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t') # Changed the delimiter to tab
        writer.writerow([f"Train Average Perplexity", f"{avg_train_perplexity:.16f}"])
        for sent, perp in zip(train_sentences, train_perplexities):
           writer.writerow([ ' '.join(sent), f"{perp:.16f}"]) # Join tokens before writing into csv

    print(f"Train Perplexity Saved to: {train_output_path}")
    
    # Calculate and save perplexity for the test set
    # Prepare test data loader
    if model_type == 'ffnn':
        test_data = prepare_ngram_data(test_sentences, word_to_index, n_gram)
        test_dataset = NgramDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, pin_memory=True, num_workers=2)
    elif model_type in ['rnn','lstm']:
        test_data = prepare_rnn_data(test_sentences, word_to_index)
        test_dataset = RNNSequenceDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2)

    test_losses, test_perplexities = evaluate_loss_per_sentence(model, test_loader, nn.NLLLoss(ignore_index=0, reduction = 'none'), model_type, device, multi_gpu=multi_gpu)
    avg_test_perplexity = np.mean(test_perplexities)

    test_output_dir = os.path.join(args.output_dir, "test_perplexity")
    os.makedirs(test_output_dir, exist_ok=True)
    
    test_output_path = os.path.join(test_output_dir, f"{args.student_id}_{model_name}_Perplexity_Test_{os.path.basename(args.corpus_path).split('.')[0]}.csv")
    
    # Save test sentences to file
    test_sentences_path = os.path.join(test_output_dir, f"{args.student_id}_test_sentences_{os.path.basename(args.corpus_path).split('.')[0]}.txt")
    with open(test_sentences_path, 'w', encoding = 'utf-8') as f:
        for sent in test_sentences:
            f.write(f"{' '.join(sent)}\n") #Join Tokens
    print(f"Test sentences saved to: {test_sentences_path}")

    with open(test_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t') # Changed the delimiter to tab
        writer.writerow([f"Test Average Perplexity", f"{avg_test_perplexity:.16f}"])
        for sent, perp in zip(test_sentences, test_perplexities):
           writer.writerow([' '.join(sent), f"{perp:.16f}"]) #Join Tokens
    
    print(f"Test Perplexity Saved to: {test_output_path}")
    

    #After training save the model
    save_path = os.path.join(args.model_dir, f'best_{model_name}_final_model.pth')
    if model_type == 'ffnn':
        save_model(model, save_path, vocab, word_to_index, model_type, best_params, n_gram)
    else:
        save_model(model, save_path, vocab, word_to_index, model_type, best_params)

    return model

def plot_loss_curves(train_losses, val_losses, model_name, model_dir, corpus_name):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.legend()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plot_path = os.path.join(model_dir, f"{model_name}_training_plot_{corpus_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plot saved to {plot_path}")

def calculate_perplexity(model, data_loader, criterion, model_type, device, multi_gpu=False):
    """Calculates perplexity using the best saved model."""
    avg_loss = evaluate_loss(model, data_loader, criterion, model_type, device, multi_gpu)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def tune_hyperparameters(text_path, model_type, data_preparation_func, dataset_class, collate_fn=None, n_gram_size=None, n_trials=10, seed = SEED, args = None):
    """Tunes hyperparameters using Optuna."""
    pp_text = load_data(text_path)
    pp_sentences = preprocess_text(pp_text)
    pp_train_sentences, pp_test_sentences = create_train_test_split(pp_sentences, test_size = 1000, seed = seed)
    pp_val_sentences = pp_train_sentences[:500]
    pp_train_sentences = pp_train_sentences[500:]
    pp_vocab, pp_word_to_index, _ = build_vocabulary(pp_train_sentences)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    multi_gpu = torch.cuda.device_count() > 1


    def objective(trial):
        embedding_dim = trial.suggest_int('embedding_dim', 8, 64)
        hidden_dim = trial.suggest_int('hidden_dim', 60, 200)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
        num_rnn_layers = trial.suggest_int('num_rnn_layers', 1, 3) if model_type in ['rnn', 'lstm'] else None
        epochs_tune = 2 # Reduced epochs for tuning

        if model_type == 'ffnn':
            context_size = n_gram_size - 1
            model = FFNNLM(len(pp_vocab), embedding_dim, context_size, hidden_dim, dropout_prob).to(device) # Move Model to device here
            train_data = data_preparation_func(pp_train_sentences, pp_word_to_index, n_gram_size)
            val_data = data_preparation_func(pp_val_sentences, pp_word_to_index, n_gram_size)
            train_dataset = dataset_class(train_data)
            val_dataset = dataset_class(val_data)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=2)

        elif model_type == 'rnn':
            model = RNNLM(len(pp_vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers).to(device) # Move Model to device here
            train_data = data_preparation_func(pp_train_sentences, pp_word_to_index)
            val_data = data_preparation_func(pp_val_sentences, pp_word_to_index)
            train_dataset = dataset_class(train_data)
            val_dataset = dataset_class(val_data)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=2, collate_fn=collate_fn)

        elif model_type == 'lstm':
            model = LSTMLM(len(pp_vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers).to(device) # Move Model to device here
            train_data = data_preparation_func(pp_train_sentences, pp_word_to_index)
            val_data = data_preparation_func(pp_val_sentences, pp_word_to_index)
            train_dataset = dataset_class(train_data)
            val_dataset = dataset_class(val_data)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=2, collate_fn=collate_fn)
        else:
            raise ValueError("Invalid model_type")

        trained_model = train_model(model, train_loader, val_loader, epochs=epochs_tune, learning_rate=learning_rate, model_type=model_type, model_name=f'tune_{model_type}', device=device, multi_gpu=multi_gpu, vocab=pp_vocab, word_to_index=pp_word_to_index, best_params = {'embedding_dim':embedding_dim, 'hidden_dim':hidden_dim, 'learning_rate':learning_rate, 'dropout_prob':dropout_prob, 'num_rnn_layers':num_rnn_layers} if model_type in ['rnn','lstm'] else {'embedding_dim':embedding_dim, 'hidden_dim':hidden_dim, 'learning_rate':learning_rate, 'dropout_prob':dropout_prob}, n_gram=n_gram_size if model_type == 'ffnn' else None, args = args, train_sentences = pp_train_sentences, test_sentences = pp_test_sentences) # Reduced epochs for tuning
        val_loss = evaluate_loss(trained_model, val_loader, nn.NLLLoss(ignore_index=0), model_type, device, multi_gpu=multi_gpu)
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best hyperparameters for {model_type}: {best_params}")
    return best_params, pp_vocab, pp_word_to_index

# --- 7. Main Execution ---
if __name__ == '__main__':
    # --- Data Loading and Preprocessing ---
    parser = argparse.ArgumentParser(description="Train Neural Language Model")
    
    parser.add_argument("--lm_type", type=str, required=True, choices=["ffnn", "rnn", "lstm"], help="Type of language model to train")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus")
    
    # Optimized Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--embedding_dim", type=int, default=80, help="Embedding dimension (was 100, optimized for generalization)")
    parser.add_argument("--hidden_dim", type=int, default=250, help="Hidden dimension (was 256, optimized to prevent overfitting)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size (was 128, smaller batches for more stable training)")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence Length for RNN and LSTM")
    parser.add_argument("--n_gram", type=int, default=3, help="N-gram size for FFNN model")
    parser.add_argument("--min_freq", type=int, default=4, help="Minimum frequency for vocab (was 1, filtering rare words helps generalization)")
    parser.add_argument("--lr", type=float, default=0.004, help="Learning rate (adjusted for better convergence)")

    # Regularization and Optimization
    parser.add_argument("--dropout", type=float, default=0.8, help="Dropout probability (increased for better regularization)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization (helps prevent overfitting)")
    
    # Model and Output Directories
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output files")
    parser.add_argument("--student_id", type=str, default="2024701027", help="Your student ID for file naming")

    # Early Stopping
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping (prevents overtraining)")

    args = parser.parse_args()

    pride_prejudice_path = args.corpus_path     
    batch_size= args.batch_size

    # --- Hyperparameter Tuning ---
    n_trials_optuna = 10 # Reduced trials for example run, increase for thorough tuning
    epochs_final_train = args.epochs

    # --- Re-train models with best hyperparameters and evaluate ---
    
    # --- FFNN ---
    if args.lm_type == 'ffnn':
      ffnn_N_gram_best_params, pp_vocab_ffnn_N_gram_, pp_word_to_index_ffnn_N_gram_ = tune_hyperparameters(
          pride_prejudice_path, 'ffnn', prepare_ngram_data, NgramDataset, n_gram_size=args.n_gram, n_trials=n_trials_optuna, seed = SEED, args = args)
      pp_ffnn_N_gram_model = FFNNLM(len(pp_vocab_ffnn_N_gram_), ffnn_N_gram_best_params['embedding_dim'], args.n_gram - 1, ffnn_N_gram_best_params['hidden_dim'], dropout_prob=ffnn_N_gram_best_params['dropout_prob']).to(device)
      pp_text = load_data(args.corpus_path)
      pp_sentences = preprocess_text(pp_text)
      pp_train_sentences, pp_test_sentences = create_train_test_split(pp_sentences, test_size = 1000, seed=SEED)
      pp_val_sentences = pp_train_sentences[:500]
      pp_train_sentences = pp_train_sentences[500:]
      pp_ngrams_3_train = prepare_ngram_data(pp_train_sentences, pp_word_to_index_ffnn_N_gram_, args.n_gram)
      pp_ngrams_3_val = prepare_ngram_data(pp_val_sentences, pp_word_to_index_ffnn_N_gram_, args.n_gram)

      
      train_dataset_ffnn_N_gram_pp = NgramDataset(pp_ngrams_3_train)
      val_dataset_ffnn_N_gram_pp = NgramDataset(pp_ngrams_3_val)
      
      train_loader_ffnn_N_gram_pp = DataLoader(train_dataset_ffnn_N_gram_pp, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
      val_loader_ffnn_N_gram_pp = DataLoader(val_dataset_ffnn_N_gram_pp, batch_size=batch_size, pin_memory=True, num_workers=2)
      
      # Prepare test data loader
      pp_ngrams_3_test = prepare_ngram_data(pp_test_sentences, pp_word_to_index_ffnn_N_gram_, args.n_gram)
      test_dataset_ffnn_N_gram_pp = NgramDataset(pp_ngrams_3_test)
      test_loader_ffnn_N_gram_pp = DataLoader(test_dataset_ffnn_N_gram_pp, batch_size=batch_size, pin_memory=True, num_workers=2)

      trained_pp_ffnn_N_gram_model = train_model(pp_ffnn_N_gram_model, train_loader_ffnn_N_gram_pp, val_loader_ffnn_N_gram_pp, epochs_final_train, model_type='ffnn', model_name = f"ffnn_{args.n_gram}-gram", learning_rate=ffnn_N_gram_best_params['learning_rate'], patience=3, device=device, multi_gpu=multi_gpu, vocab=pp_vocab_ffnn_N_gram_, word_to_index=pp_word_to_index_ffnn_N_gram_, best_params=ffnn_N_gram_best_params, n_gram = args.n_gram, args = args, train_sentences = pp_train_sentences, test_sentences = pp_test_sentences)
    
    
    elif args.lm_type == 'rnn':
      rnn_best_params, pp_vocab_rnn, pp_word_to_index_rnn = tune_hyperparameters(
          pride_prejudice_path, 'rnn', prepare_rnn_data, RNNSequenceDataset, collate_fn=collate_fn_rnn, n_trials=n_trials_optuna, seed = SEED, args = args)
      pp_rnn_model = RNNLM(len(pp_vocab_rnn), rnn_best_params['embedding_dim'], rnn_best_params['hidden_dim'], dropout_prob=rnn_best_params['dropout_prob'], num_layers = rnn_best_params['num_rnn_layers'] if 'num_rnn_layers' in rnn_best_params else 2).to(device)
      pp_text = load_data(args.corpus_path)
      pp_sentences = preprocess_text(pp_text)
      pp_train_sentences, pp_test_sentences = create_train_test_split(pp_sentences, test_size = 1000, seed=SEED)
      pp_val_sentences = pp_train_sentences[:500]
      pp_train_sentences = pp_train_sentences[500:]
      pp_rnn_sequences_train = prepare_rnn_data(pp_train_sentences, pp_word_to_index_rnn)
      pp_rnn_sequences_val = prepare_rnn_data(pp_val_sentences, pp_word_to_index_rnn)
      
      train_dataset_rnn_pp = RNNSequenceDataset(pp_rnn_sequences_train)
      val_dataset_rnn_pp = RNNSequenceDataset(pp_rnn_sequences_val)

      train_loader_rnn_pp = DataLoader(train_dataset_rnn_pp, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
      val_loader_rnn_pp = DataLoader(val_dataset_rnn_pp, batch_size=batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
      # Prepare test data loader
      pp_rnn_sequences_test = prepare_rnn_data(pp_test_sentences, pp_word_to_index_rnn)
      test_dataset_rnn_pp = RNNSequenceDataset(pp_rnn_sequences_test)
      test_loader_rnn_pp = DataLoader(test_dataset_rnn_pp, batch_size=batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2)

      trained_pp_rnn_model = train_model(pp_rnn_model, train_loader_rnn_pp, val_loader_rnn_pp, epochs_final_train, model_type='rnn', model_name='rnn', learning_rate=rnn_best_params['learning_rate'], patience=3, device=device, multi_gpu=multi_gpu, vocab = pp_vocab_rnn, word_to_index=pp_word_to_index_rnn, best_params=rnn_best_params, args = args, train_sentences = pp_train_sentences, test_sentences = pp_test_sentences)

    elif args.lm_type == 'lstm':
        lstm_best_params, pp_vocab_lstm, pp_word_to_index_lstm = tune_hyperparameters(
            pride_prejudice_path, 'lstm', prepare_rnn_data, RNNSequenceDataset, collate_fn=collate_fn_rnn, n_trials=n_trials_optuna, seed=SEED, args = args)
        # --- LSTM ---
        pp_lstm_model = LSTMLM(len(pp_vocab_lstm), lstm_best_params['embedding_dim'], lstm_best_params['hidden_dim'], dropout_prob=lstm_best_params['dropout_prob'], num_layers=lstm_best_params['num_rnn_layers'] if 'num_rnn_layers' in lstm_best_params else 2).to(device)
        pp_text = load_data(args.corpus_path)
        pp_sentences = preprocess_text(pp_text)
        pp_train_sentences, pp_test_sentences = create_train_test_split(pp_sentences, test_size = 1000, seed=SEED)
        pp_val_sentences = pp_train_sentences[:500]
        pp_train_sentences = pp_train_sentences[500:]
        pp_rnn_sequences_train = prepare_rnn_data(pp_train_sentences, pp_word_to_index_lstm)
        pp_rnn_sequences_val = prepare_rnn_data(pp_val_sentences, pp_word_to_index_lstm)
        
        train_dataset_lstm_pp = RNNSequenceDataset(pp_rnn_sequences_train)
        val_dataset_lstm_pp = RNNSequenceDataset(pp_rnn_sequences_val)

        train_loader_lstm_pp = DataLoader(train_dataset_lstm_pp, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
        val_loader_lstm_pp = DataLoader(val_dataset_lstm_pp, batch_size=batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
         # Prepare test data loader
        pp_rnn_sequences_test = prepare_rnn_data(pp_test_sentences, pp_word_to_index_lstm)
        test_dataset_lstm_pp = RNNSequenceDataset(pp_rnn_sequences_test)
        test_loader_lstm_pp = DataLoader(test_dataset_lstm_pp, batch_size=batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2)

        trained_pp_lstm_model = train_model(pp_lstm_model, train_loader_lstm_pp, val_loader_lstm_pp, epochs_final_train, model_type='lstm', model_name='lstm', learning_rate=lstm_best_params['learning_rate'], patience=3, device=device, multi_gpu=multi_gpu, vocab = pp_vocab_lstm, word_to_index = pp_word_to_index_lstm, best_params = lstm_best_params, args = args, train_sentences = pp_train_sentences, test_sentences = pp_test_sentences)

# --- Final Training and Evaluation ---    
    if args.lm_type == 'ffnn':
        pp_test_perplexity_ffnn_N_gram_ = calculate_perplexity(trained_pp_ffnn_N_gram_model, test_loader_ffnn_N_gram_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='ffnn', device=device)
        pp_train_perplexity_ffnn_N_gram_ = calculate_perplexity(trained_pp_ffnn_N_gram_model, train_loader_ffnn_N_gram_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='ffnn', device=device)
        plot_loss_curves(trained_pp_ffnn_N_gram_model.train_losses, trained_pp_ffnn_N_gram_model.val_losses, f"ffnn_{args.n_gram}-gram", args.model_dir, os.path.basename(args.corpus_path).split('.')[0])
        print(f"{os.path.basename(args.corpus_path)} FFNN ({args.n_gram}-gram) - Best Params {ffnn_N_gram_best_params}")
        print(f"{os.path.basename(args.corpus_path)} FFNN ({args.n_gram}-gram) - Test Perplexity: {pp_test_perplexity_ffnn_N_gram_:.4f}")
        print(f"{os.path.basename(args.corpus_path)} FFNN ({args.n_gram}-gram) - Train Perplexity: {pp_train_perplexity_ffnn_N_gram_:.4f}")
      
    elif args.lm_type == 'rnn':
        pp_test_perplexity_rnn = calculate_perplexity(trained_pp_rnn_model, test_loader_rnn_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='rnn', device=device)
        pp_train_perplexity_rnn = calculate_perplexity(trained_pp_rnn_model, train_loader_rnn_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='rnn', device=device)
        plot_loss_curves(trained_pp_rnn_model.train_losses, trained_pp_rnn_model.val_losses, 'rnn', args.model_dir, os.path.basename(args.corpus_path).split('.')[0])
        print(f"{os.path.basename(args.corpus_path)} RNN - Best Params {rnn_best_params}")
        print(f"{os.path.basename(args.corpus_path)} RNN - Test Perplexity: {pp_test_perplexity_rnn:.4f}")
        print(f"{os.path.basename(args.corpus_path)} RNN - Train Perplexity: {pp_train_perplexity_rnn:.4f}")
        
    elif args.lm_type == 'lstm':
        pp_test_perplexity_lstm = calculate_perplexity(trained_pp_lstm_model, test_loader_lstm_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='lstm', device=device)
        pp_train_perplexity_lstm = calculate_perplexity(trained_pp_lstm_model, train_loader_lstm_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='lstm', device=device)
        plot_loss_curves(trained_pp_lstm_model.train_losses, trained_pp_lstm_model.val_losses, 'lstm', args.model_dir, os.path.basename(args.corpus_path).split('.')[0])
        print(f"{os.path.basename(args.corpus_path)} LSTM - Best Params {lstm_best_params}")
        print(f"{os.path.basename(args.corpus_path)} LSTM - Test Perplexity: {pp_test_perplexity_lstm:.4f}")
        print(f"{os.path.basename(args.corpus_path)} LSTM - Train Perplexity: {pp_train_perplexity_lstm:.4f}")