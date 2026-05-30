"""
Bi-LSTM Temporal Encoder (BTE)

Two-layer bidirectional LSTM with temporal attention, as described in
Section 4.2 of the paper (Equations 6–8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Additive temporal attention over the Bi-LSTM hidden sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_a = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : Tensor of shape (batch, seq_len, hidden_dim)
            Concatenated bidirectional hidden states.

        Returns
        -------
        context : Tensor of shape (batch, hidden_dim)
        alpha   : Tensor of shape (batch, seq_len)   attention weights
        """
        # Eq. 7: alpha_t = softmax(W_a · tanh(W_h · h_t + b_h) + b_a)
        score = self.W_a(torch.tanh(self.W_h(h)))          # (B, T, 1)
        alpha = F.softmax(score, dim=1)                     # (B, T, 1)
        # Eq. 8: c = sum_t alpha_t * h_t
        context = (alpha * h).sum(dim=1)                    # (B, H)
        return context, alpha.squeeze(-1)


class BiLSTMEncoder(nn.Module):
    """
    Bi-LSTM Temporal Encoder (BTE).

    Architecture (Section 4.2):
        Input  → 2-layer Bi-LSTM → Temporal Attention → FC decoder → ŷ_LSTM

    Parameters
    ----------
    input_dim : int
        Number of input features per time step. Default: 13 (8 raw + 5 BFAL).
    hidden_dims : list[int]
        Hidden state sizes for the two Bi-LSTM layers. Default: [128, 64].
    forecast_horizon : int
        Number of steps to predict. Default: 1.
    dropout : float
        Dropout rate applied in decoder. Default: 0.30.
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dims: list[int] = None,
        forecast_horizon: int = 1,
        dropout: float = 0.30,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.forecast_horizon = forecast_horizon

        # Layer 1: Bi-LSTM, output dim = 2*H1
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # Layer 2: Bi-LSTM, output dim = 2*H2
        self.lstm2 = nn.LSTM(
            input_size=2 * hidden_dims[0],
            hidden_size=hidden_dims[1],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = TemporalAttention(hidden_dim=2 * hidden_dims[1])

        # Fully-connected decoder: 2*H2 → 256 → 128 → K
        attended_dim = 2 * hidden_dims[1]
        self.decoder = nn.Sequential(
            nn.Linear(attended_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, forecast_horizon),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for all weight matrices."""
        for name, param in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_dim)

        Returns
        -------
        y_hat : Tensor (batch, forecast_horizon)   primary prediction
        context : Tensor (batch, 2*H2)             attended context vector
        alpha   : Tensor (batch, seq_len)          attention weights
        """
        # Eq. 6: h_t = [h_t→ ; h_t←]
        out1, _ = self.lstm1(x)                    # (B, T, 2*H1)
        out2, _ = self.lstm2(out1)                 # (B, T, 2*H2)

        context, alpha = self.attention(out2)      # (B, 2*H2), (B, T)
        y_hat = self.decoder(context)              # (B, K)

        return y_hat, context, alpha
