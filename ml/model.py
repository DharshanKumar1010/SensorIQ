"""
Shared model definition imported by train_autoencoder.py, evaluate_autoencoder.py,
and app/services/anomaly.py.  Keep this file import-time safe (no heavy side effects).
"""
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder for time-series anomaly detection.

    Encoder compresses a window of sensor readings into a latent vector (the last
    hidden state of the top LSTM layer).  Decoder reconstructs the full sequence by
    repeating the latent vector across time and passing it through a second LSTM.
    Anomaly score = MSE between input window and reconstruction.
    """

    def __init__(
        self,
        input_size: int = 21,
        hidden_size: int = 128,
        num_layers: int = 2,
        seq_len: int = 30,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.encoder_lstm(x)          # h_n: (num_layers, batch, hidden)
        latent = h_n[-1]                              # (batch, hidden_size)
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden)
        decoder_out, _ = self.decoder_lstm(decoder_input)               # (batch, seq_len, hidden)
        return self.output_layer(decoder_out)         # (batch, seq_len, input_size)
