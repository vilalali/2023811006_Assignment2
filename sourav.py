import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import numpy as np
import random
import re # Import regular expression module
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import os
import pickle  # Import pickle for saving vocabulary

# --- 1. Setup and Data Loading ---
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

# def preprocess_text(text):
#     """Tokenizes text into sentences and words, lowercasing words."""
#     sentences = sent_tokenize(text)
#     tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]
#     return tokenized_sentences


def preprocess_text(text):
    """Tokenizes text into sentences and words, lowercasing words and removing special characters."""
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
        # Remove age values (e.g., "20 years old", "5 months old", "10-year-old") - simplified pattern
        s = re.sub(r'\d+\s*(?:year|years|month|months|day|days)\s*old', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\d+-year-old', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\d+\s*yo', '', s, flags=re.IGNORECASE) # e.g., "20 yo"

        # Remove time expressions (e.g., "12:00 am", "3:45PM", "5 seconds", "2 hours") - simplified pattern
        s = re.sub(r'\d+:\d+(?::\d+)?\s*(?:am|pm|AM|PM)', '', s)
        s = re.sub(r'\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)', '', s, flags=re.IGNORECASE)
        s = re.sub(r'(?:last|next|this)\s*(?:week|month|year)', '', s, flags=re.IGNORECASE) # Time periods

        processed_sentences.append(s)

    tokenized_sentences = [word_tokenize(s.lower()) for s in processed_sentences if s.strip()] # Tokenize and lowercase, remove empty sentences
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

    vocab = ['<PAD>', '<UNK>'] # PAD token at index 0
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

def load_vocab(vocab_path):
    """Loads vocabulary data from a file."""
    with open(vocab_path, 'rb') as f:
        vocab, word_to_index, index_to_word = pickle.load(f)
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
            sequences.append((input_seq, target_word)) # Removed extra wrapping here
    return sequences

# --- 3. Dataset Classes ---

class NgramDataset(Dataset):
    def _init_(self, ngrams):
        self.ngrams = ngrams

    def _len_(self):
        return len(self.ngrams)

    def _getitem_(self, idx):
        context, target = self.ngrams[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class RNNSequenceDataset(Dataset):
    def _init_(self, sequences):
        self.sequences = sequences

    def _len_(self):
        return len(self.sequences)

    def _getitem_(self, idx):
        input_seq, target_word = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_word, dtype=torch.long)


# --- 4. Model Definitions ---

class FFNNLM(nn.Module):
    def _init_(self, vocab_size, embedding_dim, context_size, hidden_dim, dropout_prob=0.2):
        super(FFNNLM, self)._init_()
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
    def _init_(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2): # num_layers=2
        super(RNNLM, self)._init_()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers) # num_layers=2
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

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device) # Initialize hidden for num_layers

class LSTMLM(nn.Module):
    def _init_(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2): # num_layers=2
        super(LSTMLM, self)._init_()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers) # num_layers=2
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

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device), # Initialize hidden for num_layers
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)) # Initialize cell state for num_layers

# --- 5. Training and Evaluation Functions ---

def evaluate_loss(model, data_loader, criterion, model_type, device):
    """Evaluates the model on the given data loader and returns the average loss."""
    model.eval()
    total_loss = 0
    word_count = 0
    if data_loader is None: # Handle case where val_loader is None
        return 0.0 # Return 0 loss if no validation data
    with torch.no_grad():
        for batch in data_loader:
            if model_type == 'ffnn':
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
    """Generic training function for FFNN, RNN, and LSTM models."""
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # AdamW Optimizer
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
            if model_type == 'ffnn':
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
                 avg_loss_batch = total_loss / (batch_idx+1)
                 print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Training Batch Loss: {avg_loss_batch:.4f}')


        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = 0 # Initialize val_loss to 0 in case of no val_loader
        if val_loader is not None: # Check if val_loader is provided
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
        print(f"Best Validation Loss: {best_val_loss:.4f}. Loading best model from {path_to_best_model}")
        model.load_state_dict(torch.load(path_to_best_model))

    # Plotting losses
    if val_loader is not None:
        plot_loss_curves(train_losses, val_losses, model_name)
    else:
        plot_loss_curves(train_losses, [], model_name) # Pass empty list for val_losses


    return model

def plot_loss_curves(train_losses, val_losses, model_name):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses: # Only plot validation loss if it exists
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
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

            if model_type == 'ffnn':
                if n_gram_size is None:
                    raise ValueError("n_gram_size must be specified for FFNN model")
                if len(indexed_sentence) < n_gram_size:
                    continue # Skip sentences too short for n-gram

                sentence_ngrams = []
                for i in range(n_gram_size - 1, len(indexed_sentence)):
                    context = indexed_sentence[i - (n_gram_size - 1):i]
                    target = indexed_sentence[i]
                    sentence_ngrams.append((context, target)) # Removed extra wrapping here

                if not sentence_ngrams: # Handle cases where no ngrams can be formed
                    continue

                sentence_dataset = NgramDataset(sentence_ngrams)
                sentence_dataloader = DataLoader(sentence_dataset, batch_size=1) # Batch size 1 for sentence level

                sentence_loss = 0.0
                sentence_word_count = 0
                for batch in sentence_dataloader: # Should only iterate once as batch size is 1 and dataset is for one sentence
                    contexts, targets = batch
                    contexts, targets = contexts.to(device), targets.to(device)
                    outputs = model(contexts)
                    loss = criterion(outputs, targets)
                    sentence_loss += loss.item() * targets.size(0)
                    sentence_word_count += targets.size(0)
                avg_sentence_loss = sentence_loss / sentence_word_count if sentence_word_count > 0 else 0


            elif model_type in ['rnn', 'lstm']:
                if len(indexed_sentence) < 2: # Need at least 2 words for RNN sequence
                    continue

                sentence_sequences = []
                for i in range(1, len(indexed_sentence)):
                    input_seq = indexed_sentence[:i]
                    target_word = indexed_sentence[i]
                    sentence_sequences.append((input_seq, target_word)) # Removed extra wrapping here

                if not sentence_sequences: # Handle cases where no sequence can be formed
                    continue
                sentence_dataset = RNNSequenceDataset(sentence_sequences)
                sentence_dataloader = DataLoader(sentence_dataset, batch_size=1, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch]))) # Batch size 1, collate fn

                sentence_loss = 0.0
                sentence_word_count = 0
                for batch in sentence_dataloader: # Should only iterate once
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

            sentence_perplexity = torch.exp(torch.tensor(avg_sentence_loss)).item() if avg_sentence_loss > 0 else float('inf') # Perplexity is inf for 0 loss
            sentence_perplexities.append((sentence, sentence_perplexity))
            total_loss += sentence_loss
            word_count += sentence_word_count


    avg_perplexity = torch.exp(torch.tensor(total_loss / word_count)).item() if word_count > 0 else float('inf')
    return sentence_perplexities, avg_perplexity


def write_perplexity_to_file(filename, model_name, dataset_type, sentence_perplexities):
    """Writes sentence perplexities to a file."""
    filepath = f"{filename}{model_name}{dataset_type}_perplexity.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Perplexity per sentence for {model_name} model on {dataset_type} set:\n")
        for sentence_tokens, perplexity in sentence_perplexities:
            sentence_text = " ".join(sentence_tokens)
            f.write(f"Sentence: {sentence_text}\n")
            f.write(f"Perplexity: {perplexity:.4f}\n")
            f.write("-" * 50 + "\n")
    print(f"Perplexity per sentence written to: {filepath}")

# --- 6. Prediction Functions ---

def predict_next_words_ffnn(model, context_sentence, word_to_index, index_to_word, n_gram_size, num_words_to_predict, device):
    model.eval()
    predicted_words = []
    current_context = [word.lower() for word in word_tokenize(context_sentence)]

    with torch.no_grad():
        for _ in range(num_words_to_predict):
            context_indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in current_context[- (n_gram_size - 1):]]
            if len(context_indices) < (n_gram_size - 1):
                context_indices = [word_to_index['<PAD>']] * ((n_gram_size - 1) - len(context_indices)) + context_indices # Pad if context is shorter

            context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
            output = model(context_tensor)
            probabilities = torch.exp(output)
            top_prob, top_index = torch.topk(probabilities, k=5, dim=1) # Get top 5 probabilities and indices

            next_word_index = top_index[0, 0].item() # Take the most probable word (greedy)
            next_word = index_to_word[next_word_index]
            predicted_words.append((next_word, top_prob[0,0].item())) # Store word and probability
            current_context.append(next_word) # Update context with predicted word

    return predicted_words

def predict_next_words_rnn_lstm(model, context_sentence, word_to_index, index_to_word, num_words_to_predict, model_type, device):
    model.eval()
    predicted_words = []
    current_context_tokens = [word.lower() for word in word_tokenize(context_sentence)]
    current_context_indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in current_context_tokens]

    hidden = model.init_hidden(1) # Batch size 1 for single sentence prediction

    with torch.no_grad():
        input_sequence_tensor = torch.tensor([current_context_indices], dtype=torch.long).to(device)
        for _ in range(num_words_to_predict):
            output_logprobs, hidden = model(input_sequence_tensor, hidden)
            probabilities = torch.exp(output_logprobs)
            top_prob, top_index = torch.topk(probabilities, k=5, dim=1) # Get top 5 probabilities and indices

            next_word_index = top_index[0, 0].item() # Take the most probable word (greedy)
            next_word = index_to_word[next_word_index]
            predicted_words.append((next_word, top_prob[0,0].item())) # Store word and probability

            current_context_indices.append(next_word_index)
            input_sequence_tensor = torch.tensor([[next_word_index]], dtype=torch.long).to(device) # Input for next step is just the last predicted word

    return predicted_words

def predict_next_words(input_sentence, num_words, vocab, word_to_index, index_to_word, device):
    n_gram_size_3 = 3
    n_gram_size_5 = 5
    embedding_dim = 80
    hidden_dim = 256
    dropout_prob = 0.5
    num_rnn_layers = 2


    # --- Load Trained Models ---
    pp_ffnn_3gram_model = FFNNLM(len(vocab), embedding_dim, n_gram_size_3 - 1, hidden_dim, dropout_prob).to(device)
    pp_ffnn_5gram_model = FFNNLM(len(vocab), embedding_dim, n_gram_size_5 - 1, hidden_dim, dropout_prob).to(device)
    pp_rnn_model = RNNLM(len(vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers).to(device)
    pp_lstm_model = LSTMLM(len(vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers).to(device)

    pp_ffnn_3gram_model.load_state_dict(torch.load('best_ffnn_3gram_model.pth', map_location=device))
    pp_ffnn_5gram_model.load_state_dict(torch.load('best_ffnn_5gram_model.pth', map_location=device))
    pp_rnn_model.load_state_dict(torch.load('best_rnn_model.pth', map_location=device))
    pp_lstm_model.load_state_dict(torch.load('best_lstm_model.pth', map_location=device))

    print(f"\n--- Predictions from FFNN (3-gram) Model ---")
    predicted_ffnn_3gram = predict_next_words_ffnn(pp_ffnn_3gram_model, input_sentence, word_to_index, index_to_word, n_gram_size_3, num_words, device)
    for word, prob in predicted_ffnn_3gram:
        print(f"Word: '{word}', Probability: {prob:.4f}")

    print(f"\n--- Predictions from FFNN (5-gram) Model ---")
    predicted_ffnn_5gram = predict_next_words_ffnn(pp_ffnn_5gram_model, input_sentence, word_to_index, index_to_word, n_gram_size_5, num_words, device)
    for word, prob in predicted_ffnn_5gram:
        print(f"Word: '{word}', Probability: {prob:.4f}")

    print(f"\n--- Predictions from RNN Model ---")
    predicted_rnn = predict_next_words_rnn_lstm(pp_rnn_model, input_sentence, word_to_index, index_to_word, num_words, 'rnn', device)
    for word, prob in predicted_rnn:
        print(f"Word: '{word}', Probability: {prob:.4f}")

    print(f"\n--- Predictions from LSTM Model ---")
    predicted_lstm = predict_next_words_rnn_lstm(pp_lstm_model, input_sentence, word_to_index, index_to_word, num_words, 'lstm', device)
    for word, prob in predicted_lstm:
        print(f"Word: '{word}', Probability: {prob:.4f}")


# --- 7. Main Execution ---

if _name_ == '_main_':
    # --- Data Loading and Preprocessing ---
    pride_prejudice_path = '/kaggle/input/assignment-nlp/Pride-and-Prejudice-Jane-Austen.txt' # Replace with your actual path

    pride_prejudice_text = load_data(pride_prejudice_path)
    pp_sentences = preprocess_text(pride_prejudice_text)

    pp_train_sentences, pp_val_sentences, pp_test_sentences = create_train_test_split(pp_sentences, val_size=500)
    pp_vocab, pp_word_to_index, pp_index_to_word = build_vocabulary(pp_train_sentences)

    # Save Vocabulary
    vocab_data = (pp_vocab, pp_word_to_index, pp_index_to_word)
    save_vocab(vocab_data, 'pp_vocab.pkl') # Save vocabulary to file

    print(f"Pride and Prejudice Vocabulary size: {len(pp_vocab)}")

    # --- Prepare Data for Models ---
    n_gram_size_3 = 3
    n_gram_size_5 = 5

    pp_ngrams_3_train = prepare_ngram_data(pp_train_sentences, pp_word_to_index, n_gram_size_3)
    pp_ngrams_5_train = prepare_ngram_data(pp_train_sentences, pp_word_to_index, n_gram_size_5)
    pp_ngrams_3_val = prepare_ngram_data(pp_val_sentences, pp_word_to_index, n_gram_size_3)
    pp_ngrams_5_val = prepare_ngram_data(pp_val_sentences, pp_word_to_index, n_gram_size_5)
    pp_ngrams_3_test = prepare_ngram_data(pp_test_sentences, pp_word_to_index, n_gram_size_3)
    pp_ngrams_5_test = prepare_ngram_data(pp_test_sentences, pp_word_to_index, n_gram_size_5)


    pp_rnn_sequences_train = prepare_rnn_data(pp_train_sentences, pp_word_to_index)
    pp_rnn_sequences_val = prepare_rnn_data(pp_val_sentences, pp_word_to_index)
    pp_rnn_sequences_test = prepare_rnn_data(pp_test_sentences, pp_word_to_index)


    # --- Hyperparameters ---
    embedding_dim = 80
    hidden_dim = 256
    learning_rate = 0.001
    epochs = 20  # Increased epochs to see effect of regularization
    batch_size = 32
    dropout_prob = 0.5  # Dropout probability
    num_rnn_layers = 2  # Number of RNN/LSTM layers


    # --- Initialize and Train FFNN Models (Pride & Prejudice) ---
    pp_ffnn_3gram_model = FFNNLM(len(pp_vocab), embedding_dim, n_gram_size_3 - 1, hidden_dim, dropout_prob)
    pp_ffnn_5gram_model = FFNNLM(len(pp_vocab), embedding_dim, n_gram_size_5 - 1, hidden_dim, dropout_prob)

    train_dataset_ffnn_3gram_pp = NgramDataset(pp_ngrams_3_train)
    val_dataset_ffnn_3gram_pp = NgramDataset(pp_ngrams_3_val)
    train_loader_ffnn_3gram_pp = DataLoader(train_dataset_ffnn_3gram_pp, batch_size=batch_size, shuffle=True, pin_memory=True) # pin_memory=True
    val_loader_ffnn_3gram_pp = DataLoader(val_dataset_ffnn_3gram_pp, batch_size=batch_size, pin_memory=True) # pin_memory=True

    train_dataset_ffnn_5gram_pp = NgramDataset(pp_ngrams_5_train)
    val_dataset_ffnn_5gram_pp = NgramDataset(pp_ngrams_5_val)
    train_loader_ffnn_5gram_pp = DataLoader(train_dataset_ffnn_5gram_pp, batch_size=batch_size, shuffle=True, pin_memory=True) # pin_memory=True
    val_loader_ffnn_5gram_pp = DataLoader(val_dataset_ffnn_5gram_pp, batch_size=batch_size, pin_memory=True) # pin_memory=True


    trained_pp_ffnn_3gram_model = train_model(pp_ffnn_3gram_model, train_loader_ffnn_3gram_pp, val_loader_ffnn_3gram_pp, epochs, learning_rate, model_type='ffnn', model_name='ffnn_3gram')
    print("FFNN (3-gram) model for Pride & Prejudice trained.")
    trained_pp_ffnn_5gram_model = train_model(pp_ffnn_5gram_model, train_loader_ffnn_5gram_pp, val_loader_ffnn_5gram_pp, epochs, learning_rate, model_type='ffnn', model_name='ffnn_5gram')
    print("FFNN (5-gram) model for Pride & Prejudice trained.")


    # --- Initialize and Train RNN Model (Pride & Prejudice) ---
    pp_rnn_model = RNNLM(len(pp_vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers) # num_layers
    train_dataset_rnn_pp = RNNSequenceDataset(pp_rnn_sequences_train)
    val_dataset_rnn_pp = RNNSequenceDataset(pp_rnn_sequences_val)
    train_loader_rnn_pp = DataLoader(train_dataset_rnn_pp, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True) # pin_memory=True
    val_loader_rnn_pp = DataLoader(val_dataset_rnn_pp, batch_size=batch_size, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)

    trained_pp_rnn_model = train_model(pp_rnn_model, train_loader_rnn_pp, val_loader_rnn_pp, epochs, learning_rate, model_type='rnn', model_name='rnn')
    print("RNN model for Pride & Prejudice trained.")


    # --- Initialize and Train LSTM Model (Pride & Prejudice) ---
    pp_lstm_model = LSTMLM(len(pp_vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers) # num_layers
    train_dataset_lstm_pp = RNNSequenceDataset(pp_rnn_sequences_train)
    val_dataset_lstm_pp = RNNSequenceDataset(pp_rnn_sequences_val)
    train_loader_lstm_pp = DataLoader(train_dataset_lstm_pp, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True) # pin_memory=True
    val_loader_lstm_pp = DataLoader(val_dataset_lstm_pp, batch_size=batch_size, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True)

    trained_pp_lstm_model = train_model(pp_lstm_model, train_loader_lstm_pp, val_loader_lstm_pp, epochs, learning_rate, model_type='lstm', model_name='lstm')
    print("LSTM model for Pride & Prejudice trained.")


    # --- Perplexity Calculation (Pride & Prejudice) ---
    criterion_perplexity = nn.NLLLoss(reduction='mean', ignore_index=0)

    # --- Perplexity Calculation per sentence and write to file ---
    output_filename_prefix = "pp_perplexity"

    # FFNN 3-gram
    pp_ffnn_3gram_model.load_state_dict(torch.load('best_ffnn_3gram_model.pth'))
    train_perplexity_ffnn_3gram_sentences, avg_train_perplexity_ffnn_3gram = calculate_perplexity_per_sentence(pp_ffnn_3gram_model, pp_train_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='ffnn', n_gram_size=n_gram_size_3)
    val_perplexity_ffnn_3gram_sentences, avg_val_perplexity_ffnn_3gram = calculate_perplexity_per_sentence(pp_ffnn_3gram_model, pp_val_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='ffnn', n_gram_size=n_gram_size_3)
    test_perplexity_ffnn_3gram_sentences, avg_test_perplexity_ffnn_3gram = calculate_perplexity_per_sentence(pp_ffnn_3gram_model, pp_test_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='ffnn', n_gram_size=n_gram_size_3)
    write_perplexity_to_file(output_filename_prefix, 'ffnn_3gram', 'train', train_perplexity_ffnn_3gram_sentences)
    write_perplexity_to_file(output_filename_prefix, 'ffnn_3gram', 'val', val_perplexity_ffnn_3gram_sentences)
    write_perplexity_to_file(output_filename_prefix, 'ffnn_3gram', 'test', test_perplexity_ffnn_3gram_sentences)
    print(f"Pride & Prejudice FFNN (3-gram) - Avg Train Perplexity: {avg_train_perplexity_ffnn_3gram:.4f}, Avg Val Perplexity: {avg_val_perplexity_ffnn_3gram:.4f}, Avg Test Perplexity: {avg_test_perplexity_ffnn_3gram:.4f}")


    # FFNN 5-gram
    pp_ffnn_5gram_model.load_state_dict(torch.load('best_ffnn_5gram_model.pth'))
    train_perplexity_ffnn_5gram_sentences, avg_train_perplexity_ffnn_5gram = calculate_perplexity_per_sentence(pp_ffnn_5gram_model, pp_train_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='ffnn', n_gram_size=n_gram_size_5)
    val_perplexity_ffnn_5gram_sentences, avg_val_perplexity_ffnn_5gram = calculate_perplexity_per_sentence(pp_ffnn_5gram_model, pp_val_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='ffnn', n_gram_size=n_gram_size_5)
    test_perplexity_ffnn_5gram_sentences, avg_test_perplexity_ffnn_5gram = calculate_perplexity_per_sentence(pp_ffnn_5gram_model, pp_test_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='ffnn', n_gram_size=n_gram_size_5)
    write_perplexity_to_file(output_filename_prefix, 'ffnn_5gram', 'train', train_perplexity_ffnn_5gram_sentences)
    write_perplexity_to_file(output_filename_prefix, 'ffnn_5gram', 'val', val_perplexity_ffnn_5gram_sentences)
    write_perplexity_to_file(output_filename_prefix, 'ffnn_5gram', 'test', test_perplexity_ffnn_5gram_sentences)
    print(f"Pride & Prejudice FFNN (5-gram) - Avg Train Perplexity: {avg_train_perplexity_ffnn_5gram:.4f}, Avg Val Perplexity: {avg_val_perplexity_ffnn_5gram:.4f}, Avg Test Perplexity: {avg_test_perplexity_ffnn_5gram:.4f}")

    # RNN
    pp_rnn_model.load_state_dict(torch.load('best_rnn_model.pth'))
    train_perplexity_rnn_sentences, avg_train_perplexity_rnn = calculate_perplexity_per_sentence(pp_rnn_model, pp_train_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='rnn')
    val_perplexity_rnn_sentences, avg_val_perplexity_rnn = calculate_perplexity_per_sentence(pp_rnn_model, pp_val_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='rnn')
    test_perplexity_rnn_sentences, avg_test_perplexity_rnn = calculate_perplexity_per_sentence(pp_rnn_model, pp_test_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='rnn')
    write_perplexity_to_file(output_filename_prefix, 'rnn', 'train', train_perplexity_rnn_sentences)
    write_perplexity_to_file(output_filename_prefix, 'rnn', 'val', val_perplexity_rnn_sentences)
    write_perplexity_to_file(output_filename_prefix, 'rnn', 'test', test_perplexity_rnn_sentences)
    print(f"Pride & Prejudice RNN - Avg Train Perplexity: {avg_train_perplexity_rnn:.4f}, Avg Val Perplexity: {avg_val_perplexity_rnn:.4f}, Avg Test Perplexity: {avg_test_perplexity_rnn:.4f}")


    # LSTM
    pp_lstm_model.load_state_dict(torch.load('best_lstm_model.pth'))
    train_perplexity_lstm_sentences, avg_train_perplexity_lstm = calculate_perplexity_per_sentence(pp_lstm_model, pp_train_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='lstm')
    val_perplexity_lstm_sentences, avg_val_perplexity_lstm = calculate_perplexity_per_sentence(pp_lstm_model, pp_val_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='lstm')
    test_perplexity_lstm_sentences, avg_test_perplexity_lstm = calculate_perplexity_per_sentence(pp_lstm_model, pp_test_sentences, pp_word_to_index, pp_index_to_word, criterion_perplexity, model_type='lstm')
    write_perplexity_to_file(output_filename_prefix, 'lstm', 'train', train_perplexity_lstm_sentences)
    write_perplexity_to_file(output_filename_prefix, 'lstm', 'val', val_perplexity_lstm_sentences)
    write_perplexity_to_file(output_filename_prefix, 'lstm', 'test', test_perplexity_lstm_sentences)
    print(f"Pride & Prejudice LSTM - Avg Train Perplexity: {avg_train_perplexity_lstm:.4f}, Avg Val Perplexity: {avg_val_perplexity_lstm:.4f}, Avg Test Perplexity: {avg_test_perplexity_lstm:.4f}")

    # --- Next Word Prediction ---
    input_sent = input("\n\nEnter a sentence to start prediction: ")
    num_predict_words = int(input("Enter the number of words to predict: "))

    predict_next_words(input_sent, num_predict_words, pp_vocab, pp_word_to_index, pp_index_to_word, device)


    print("="*50) # Separator
    print("Remember to train and evaluate Ulysses models as well, following the same structure!")
    print("="*50)