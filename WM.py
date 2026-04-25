import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_dim=32):
        super(VAE, self).__init__()
        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # z_dim output
        self.fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, z_dim)
        
        # Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.decoder_fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder_fc(z), self.decoder(self.decoder_fc(z)), mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(self.decoder_fc(z))

class MDNRNN(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_dim, num_gaussians):
        super(MDNRNN, self).__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        
        # RNN takes concatenated [z, action]
        self.rnn = nn.LSTM(z_dim + action_dim, hidden_dim, batch_first=True)
        
        # Prediction for next z: pi, mu, sigma
        self.fc = nn.Linear(hidden_dim, num_gaussians * z_dim * 3)

    def forward(self, z, action, hidden=None):
        # z: (batch, seq, z_dim), action: (batch, seq, action_dim)
        x = torch.cat([z, action], dim=-1)
        output, hidden = self.rnn(x, hidden)
        
        # Parameters for next z
        params = self.fc(output)
        
        # Reshape to (batch, seq, num_gaussians, 3, z_dim)
        bs, seq_len, _ = output.shape
        params = params.view(bs, seq_len, self.num_gaussians, 3, self.z_dim)
        
        pi = F.softmax(params[:, :, :, 0, 0], dim=2) # Probabilities over mixtures (shared across z_dim)
        mu = params[:, :, :, 1, :]
        sigma = torch.exp(params[:, :, :, 2, :])
        
        return pi, mu, sigma, hidden

class Controller(nn.Module):
    def __init__(self, z_dim, hidden_dim, action_dim):
        super(Controller, self).__init__()
        # Simple linear controller: a = W * [z, h] + b
        self.fc = nn.Linear(z_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        return torch.tanh(self.fc(torch.cat([z, h], dim=-1)))

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD