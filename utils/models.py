from torch import nn
import torch

def init_weights(model):
    for m in model.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

class EmbedderNet(nn.Module):
    """
    Embedding network that maps original time series to latent space. 
    Parameters:
        input dim:
        hidden dim:
        num_layers:
        output_dim:
    Args:
        X: input time series of shape [batch, seq_len, input_dim]
    Returns:
        H: embeddings
    """
    def __init__(self, input_dim=5, hidden_dim=20, num_layers=3, output_dim=20):
        super(EmbedderNet, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())
        init_weights(self)

    def forward(self, X):
        out, h = self.lstm(X)
        H = self.fc(out)
        return H
    

class RecoveryNet(nn.Module):
    """
    Recovery network that maps latent space into original space.
    Parameters:
        input dim:
        hidden dim:
        num_layers:
        output_dim:
    Args:
        H: Latent representation of shape [batch, seq_len, latent_dim]
    Returns:
        X_tilde: decoded time series
    """
    def __init__(self, input_dim=20, hidden_dim=20, num_layers=3, output_dim=5):
        super(RecoveryNet, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())
        init_weights(self)

    def forward(self, H):
        out, h = self.lstm(H)
        X_tilde = self.fc(out)
        return X_tilde
    
class GeneratorNet(nn.Module):
    """
    Generator network that generates time-series data in the latent space.
    Parameters:
        input dim:
        hidden dim:
        num_layers:
        output_dim:
    Args:
        Z: random variables
    Returns:
        E: generated embedding
    """
    def __init__(self, input_dim=5, hidden_dim=20, num_layers=3, output_dim=20):
        super(GeneratorNet, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())
        init_weights(self)

    def forward(self, Z):
        out, h = self.lstm(Z)
        E = self.fc(out)
        return E

    
class SupervisorNet(nn.Module):
    """
    Supervisor network that generates the next sequence in latent representation from the previous sequence.
    Parameters:
        input dim:
        hidden dim:
        num_layers:
        output_dim:
    Args:
        H: Latent representation of shape [batch, seq_len, latent_dim]
    Returns:
        S: generated sequence based on the latent representations.
    """
    def __init__(self, input_dim=20, hidden_dim=20, num_layers=2, output_dim=20):
        super(SupervisorNet, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())
        init_weights(self)

    def forward(self, H):
        out, h = self.lstm(H)
        S = self.fc(out)
        return S
    
    
class DiscriminatorNet(nn.Module):
    """
    Discriminator network to discriminate between the original and generated time-series.
    Parameters:
        input dim:
        hidden dim:
        num_layers:
    Args:
        H: Latent representation of shape [batch, seq_len, latent_dim]
    Returns:
        Y_hat: 1-dim classification result.
    """
    def __init__(self, input_dim=20, hidden_dim=20, num_layers=3):
        super(DiscriminatorNet, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        init_weights(self)

    def forward(self, H):
        out, h = self.lstm(H)
        Y_hat = self.fc(out)
        return Y_hat