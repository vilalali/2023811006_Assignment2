import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import pickle
import argparse
import os
from model_train import FFNNLM, RNNLM, LSTMLM
import re

def load_vocab(vocab_path):
    """Loads vocabulary data from the provided path."""
    print(vocab_path)  # Keep this for debugging
    try:
        with open(vocab_path, 'rb') as f:
            vocab, word_to_index, index_to_word = pickle.load(f)
            return vocab, word_to_index, index_to_word
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Make sure the path is correct.")
    except Exception as e: # catch other exceptions
        raise Exception(f"Error loading vocabulary from {vocab_path}: {e}")

def predict_top_n_words_ffnn(model, context_sentence, word_to_index, index_to_word, n_gram_size, top_n, device):
    """Predicts the top N most probable next words for FFNN."""
    model.eval()
    current_context = [word.lower() for word in word_tokenize(context_sentence)]
    with torch.no_grad():
        context_indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in current_context[-(n_gram_size - 1):]]
        if len(context_indices) < (n_gram_size - 1):
            context_indices = [word_to_index['<PAD>']] * ((n_gram_size - 1) - len(context_indices)) + context_indices
        context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
        output = model(context_tensor)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_n, dim=1)
        predictions = []
        for i in range(top_n):
            word = index_to_word[top_indices[0][i].item()]
            prob = top_probs[0][i].item()
            predictions.append(f"{word} {prob:.4f}")
    return predictions


def predict_top_n_words_rnn_lstm(model, context_sentence, word_to_index, index_to_word, top_n, device):
    """Predicts the top N most probable next words for RNN/LSTM."""
    model.eval()
    current_context_indices = [word_to_index.get(word.lower(), word_to_index['<UNK>']) for word in word_tokenize(context_sentence)]
    hidden = model.init_hidden(1, device)
    with torch.no_grad():
        input_sequence_tensor = torch.tensor([current_context_indices], dtype=torch.long).to(device)
        output_logprobs, hidden = model(input_sequence_tensor, hidden)
        probabilities = F.softmax(output_logprobs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_n, dim=1)
        predictions = []
        for i in range(top_n):
            word = index_to_word[top_indices[0][i].item()]
            prob = top_probs[0][i].item()
            predictions.append(f"{word} {prob:.4f}")
    return predictions


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Vocabulary directly from the provided path
    vocab, word_to_index, index_to_word = load_vocab(args.vocab_path)

    model_type = args.model_type
    n_gram_size = None

    if model_type.startswith("ffnn"):
        if "3gram" in model_type:  n_gram_size = 3
        elif "5gram" in model_type: n_gram_size = 5
        else: raise ValueError("Invalid ffnn type. Use ffnn_3gram or ffnn_5gram.")

    if model_type.startswith('ffnn'):
        model = FFNNLM(len(vocab), args.embedding_dim, n_gram_size - 1, args.hidden_dim, args.dropout_prob).to(device)
    elif model_type == 'rnn':
        model = RNNLM(len(vocab), args.embedding_dim, args.hidden_dim, args.dropout_prob, args.num_rnn_layers).to(device)
    elif model_type == 'lstm':
        model = LSTMLM(len(vocab), args.embedding_dim, args.hidden_dim, args.dropout_prob, args.num_rnn_layers).to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model.load_state_dict(torch.load(args.trained_model_path, map_location=device))
    model.eval()

    input_sentence = input("Input Sentence: ")
    print(f"Input Sentence: {input_sentence}")

    if model_type.startswith('ffnn'):
        predictions = predict_top_n_words_ffnn(model, input_sentence, word_to_index, index_to_word, n_gram_size, args.num_words, device)
    else:
        predictions = predict_top_n_words_rnn_lstm(model, input_sentence, word_to_index, index_to_word, args.num_words, device)

    print("Output:")
    for pred in predictions:
        print(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict next words.')
    parser.add_argument('--num_words', type=int, required=True, help='Number of top words to predict.')
    parser.add_argument('--trained_model_path', type=str, required=True, help='Path to trained model.')
    parser.add_argument('--model_type', type=str, required=True, choices=['ffnn_3gram', 'ffnn_5gram', 'rnn', 'lstm'], help='Model type.')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary file (vocab.pkl).')  # Changed argument name
    parser.add_argument('--embedding_dim', type=int, default=80, help='Embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension.')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability.')
    parser.add_argument('--num_rnn_layers', type=int, default=2, help='Number of RNN/LSTM layers.')
    args = parser.parse_args()

    predict(args)