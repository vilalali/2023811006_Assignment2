import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import argparse
import os
from collections import Counter
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import sent_tokenize


# --- 1. Setup and Data Loading ---
SEED = 49
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
print(f"Using device: {device}, Number of GPUs available: {n_gpus}")
multi_gpu = n_gpus > 1

def load_data(file_path):
    """Loads text data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def preprocess_text(text):
    """Tokenizes text into sentences and words, lowercasing words."""
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]
    return tokenized_sentences

def create_train_test_split(sentences, val_size=500, test_size=500, seed = SEED):
    """Splits sentences into training, validation, and test sets."""
    random.seed(seed)
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

# --- 4. Model Definitions ---

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

def load_model(model_path):
    """Loads a saved model from the given path."""
    state = torch.load(model_path, map_location=device)
    model_type = state['model_type']
    vocab = state['vocab']
    word_to_index = state['word_to_index']
    best_params = state['best_params']
    n_gram = state.get('n_gram')

    if model_type == 'ffnn':
        model = FFNNLM(len(vocab), best_params['embedding_dim'], n_gram - 1, best_params['hidden_dim'], dropout_prob=best_params['dropout_prob']).to(device)
    elif model_type == 'rnn':
       model = RNNLM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'], dropout_prob=best_params['dropout_prob'], num_layers = best_params['num_rnn_layers'] if 'num_rnn_layers' in best_params else 2).to(device)
    elif model_type == 'lstm':
        model = LSTMLM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'], dropout_prob=best_params['dropout_prob'], num_layers=best_params['num_rnn_layers'] if 'num_rnn_layers' in best_params else 2).to(device)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, vocab, word_to_index, model_type, n_gram

def predict_next_word(model, input_sentence, word_to_index, index_to_word, model_type, k, n_gram=None):
    """Predicts the next word given a model and a sentence."""
    model.eval()
    with torch.no_grad():
        tokenized_sentence = [word.lower() for word in word_tokenize(input_sentence)]
        indexed_sentence = [word_to_index.get(word, word_to_index['<UNK>']) for word in tokenized_sentence]

        if model_type == 'ffnn':
            if len(indexed_sentence) < n_gram -1 :
                return "Sentence too short for model prediction" # Handle short inputs
            context = torch.tensor(indexed_sentence[-(n_gram -1):], dtype=torch.long).unsqueeze(0).to(device)
            output = model(context)
        elif model_type in ['rnn', 'lstm']:
            input_seq = torch.tensor(indexed_sentence, dtype=torch.long).unsqueeze(0).to(device)
            if multi_gpu:
                hidden = model.module.init_hidden(input_seq.size(0), device)
            else:
                hidden = model.init_hidden(input_seq.size(0), device)
            output, _ = model(input_seq, hidden)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
        probs = torch.exp(output).squeeze().cpu()
        top_k_indices = torch.topk(probs, k).indices.tolist()
        top_k_words_probs = [(index_to_word[idx], probs[idx].item()) for idx in top_k_indices]
        return top_k_words_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text using a trained language model.")
    parser.add_argument("lm_type", type=str, choices=['ffnn', 'rnn', 'lstm'], help="Type of language model ('ffnn' for FFNN, 'rnn' for RNN, 'lstm' for LSTM)")
    parser.add_argument("corpus_path", type=str, help="Path to the corpus text file.")
    parser.add_argument("k", type=int, help="Number of candidates for the next word.")
    parser.add_argument("--model_path", type=str, required=True, help = "Path to the trained model file")
    args = parser.parse_args()

    model_type = args.lm_type
    model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit()

    model, vocab, word_to_index, model_type, n_gram = load_model(model_path)
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    while True:
        input_sentence = input("Input sentence: ")
        if input_sentence.lower() == 'exit':
            break

        try:
            next_word_predictions = predict_next_word(model, input_sentence, word_to_index, index_to_word, model_type, args.k, n_gram)
            if isinstance(next_word_predictions, str):
                print(next_word_predictions)
            else:
                for word, prob in next_word_predictions:
                    print(f"{word} {prob:.4f}")
        except Exception as e:
             print(f"An error occurred: {e}")