from tqdm import tqdm
import torch
import torch.nn as nn
from models import AutoEncoder

def train(embeddings_matrix, input_dim = 300, hidden_dim = 100, num_epochs = 50):
    autoencoder = AutoEncoder(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        reconstructed, encoded = autoencoder(embeddings_matrix)
        loss = loss_fn(reconstructed, embeddings_matrix)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    _, embeddings_transformed = autoencoder(embeddings_matrix)
    return autoencoder, embeddings_transformed, losses