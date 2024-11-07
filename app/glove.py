import torch

def load_glove_vectors(glove_file):
    glove_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            glove_vectors[word] = vector
    return glove_vectors


def get_vocabulary_from_glove(glove_vectors):
    vocab = dict()
    inverse_vocab = list()
    vocab["<PAD>"] = 0
    inverse_vocab.append("<PAD>")
    vocab["<UNK>"] = 1
    inverse_vocab.append("<UNK>")
    for word, vector in glove_vectors.items():
        vocab[word] = len(inverse_vocab)
        inverse_vocab.append(word)
    return vocab, inverse_vocab

def get_glove_embedding(word, glove_vectors):
    return glove_vectors.get(word, torch.zeros(300))

def get_sentence_embedding(sentence, glove_vectors):
    words = sentence.split()
    word_embeddings = [get_glove_embedding(word, glove_vectors) for word in words if word in glove_vectors]
    if word_embeddings:
        return torch.mean(torch.stack(word_embeddings), dim=0)
    else:
        return torch.zeros(300)