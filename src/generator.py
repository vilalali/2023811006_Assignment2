# generator.py

import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import pickle
import argparse
import os
from model_train import FFNNLM, RNNLM, LSTMLM  # Import model classes


def load_vocab(vocab_path):
    """Loads vocabulary data from a file."""
    with open(vocab_path, 'rb') as f:
        vocab, word_to_index, index_to_word = pickle.load(f)
    return vocab, word_to_index, index_to_word


def predict_next_words_ffnn(model, context_sentence, word_to_index, index_to_word, n_gram_size, num_words_to_predict, device):
    """Predicts next words using FFNN."""
    model.eval()
    predicted_words = []
    current_context = [word.lower() for word in word_tokenize(context_sentence)]

    with torch.no_grad():
        for _ in range(num_words_to_predict):
            context_indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in current_context[- (n_gram_size - 1):]]
            if len(context_indices) < (n_gram_size - 1):
                context_indices = [word_to_index['<PAD>']] * ((n_gram_size - 1) - len(context_indices)) + context_indices

            context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
            output = model(context_tensor)
            probabilities = torch.exp(output)
            top_prob, top_index = torch.topk(probabilities, k=1, dim=1)

            next_word_index = top_index[0, 0].item()
            next_word = index_to_word[next_word_index]
            predicted_words.append(next_word)
            current_context.append(next_word)
    return predicted_words


def predict_next_words_rnn_lstm(model, context_sentence, word_to_index, index_to_word, num_words_to_predict, model_type, device):
    """Predicts the next words using the RNN or LSTM model."""
    model.eval()
    predicted_words = []
    current_context_tokens = [word.lower() for word in word_tokenize(context_sentence)]
    current_context_indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in current_context_tokens]

    hidden = model.init_hidden(1)

    with torch.no_grad():
        input_sequence_tensor = torch.tensor([current_context_indices], dtype=torch.long).to(device)
        for _ in range(num_words_to_predict):
            output_logprobs, hidden = model(input_sequence_tensor, hidden)
            probabilities = torch.exp(output_logprobs)
            top_prob, top_index = torch.topk(probabilities, k=1, dim=1)

            next_word_index = top_index[0, 0].item()
            next_word = index_to_word[next_word_index]
            predicted_words.append(next_word)

            current_context_indices.append(next_word_index)
            input_sequence_tensor = torch.tensor([[next_word_index]], dtype=torch.long).to(device)

    return predicted_words


def predict(args):
    """Loads a model and vocabulary, and predicts the next words."""

    # Load Vocabulary
    vocab, word_to_index, index_to_word = load_vocab(args.vocab_path)

    # Load the specified model
    if args.model_type == 'ffnn_3gram':
        n_gram_size = 3
        model = FFNNLM(len(vocab), args.embedding_dim, n_gram_size - 1, args.hidden_dim, args.dropout_prob).to(args.device)
    elif args.model_type == 'ffnn_5gram':
        n_gram_size = 5
        model = FFNNLM(len(vocab), args.embedding_dim, n_gram_size - 1, args.hidden_dim, args.dropout_prob).to(args.device)
    elif args.model_type == 'rnn':
        model = RNNLM(len(vocab), args.embedding_dim, args.hidden_dim, args.dropout_prob, args.num_rnn_layers).to(args.device)
    elif args.model_type == 'lstm':
        model = LSTMLM(len(vocab), args.embedding_dim, args.hidden_dim, args.dropout_prob, args.num_rnn_layers).to(args.device)
    else:
        raise ValueError(f"Invalid model_type: {args.model_type}")

    model.load_state_dict(torch.load(args.trained_model_path, map_location=args.device))
    model.eval()

    # Get input sentence from the console
    input_sentence = input("Input Sentence: ")
    print(f"Input Sentence: {input_sentence}")


    # Predict next words
    if args.model_type.startswith('ffnn'):
        predictions = predict_next_words_ffnn(model, input_sentence, word_to_index, index_to_word, n_gram_size, args.num_words, args.device)
    else:  # rnn or lstm
        predictions = predict_next_words_rnn_lstm(model, input_sentence, word_to_index, index_to_word, args.num_words, args.model_type, args.device)

    # Print predictions
    print(" ".join(predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict next words using a trained language model.')
    # Removed --input_sentence argument
    parser.add_argument('--num_words', type=int, required=True,
                        help='Number of words to predict.')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path to the vocabulary file (e.g., pp_vocab.pkl).')
    parser.add_argument('--trained_model_path', type=str, required=True,
                        help='Path to the trained model file (e.g., best_pp_rnn_model.pth).')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['ffnn_3gram', 'ffnn_5gram', 'rnn', 'lstm'],
                        help='Type of model to use for prediction.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu).')
    # Hyperparameter arguments
    parser.add_argument('--embedding_dim', type=int, default=80, help='Embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension.')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability.')
    parser.add_argument('--num_rnn_layers', type=int, default=2, help='Number of RNN/LSTM layers.')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU.")
        args.device = 'cpu'
    args.device = torch.device(args.device)

    predict(args)