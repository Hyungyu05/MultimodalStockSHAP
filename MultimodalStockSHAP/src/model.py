import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Many-to-One LSTM
        Args:
            input_size: Feature 개수 (6: OHLCV + Sentiment)
            hidden_size: LSTM hidden dimension
            num_layers: LSTM layer 개수
            output_size: 출력 차원 (1: Next Day Close Price)
            dropout: Dropout 비율
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            out: (batch, 1) - Scalar prediction
        """
        # LSTM forward
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden)
        
        # Many-to-One: 마지막 타임스텝만 사용
        last_output = lstm_out[:, -1, :]  # (batch, hidden)
        out = self.fc(last_output)        # (batch, 1)
        
        return out.squeeze()  # (batch,)
