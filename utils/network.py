import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CandlestickDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature.float(), label.float()

class FeatureMaskLayer(nn.Module):
    def __init__(self, input_size: int, device: torch.device):
        super().__init__()
        self.device = device
        self.mask = torch.ones(input_size, device=device)
        self.to(device)
        
    def forward(self, x):
        return x * self.mask

class LSTMNetwork(nn.Module):
    def __init__(self, 
        input_size: int, 
        lstm_hidden_sizes: list[int], 
        fc_sizes: list[int], 
        dropout_rate: int, 
        lstm_layers: int,
        is_bidirectional: int, 
        device: torch.device
    ):
        super(LSTMNetwork, self).__init__()
        self.input_size = input_size
        self.lstm_hidden_sizes = lstm_hidden_sizes
        self.fc_sizes = fc_sizes
        self.dropout_rate = dropout_rate
        self.lstm_layers_count = lstm_layers
        self.is_bidirectional = is_bidirectional
        self.device = device

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in lstm_hidden_sizes:
            self.lstm_layers.append(nn.LSTM(prev_size, hidden_size, num_layers=lstm_layers, dropout=dropout_rate, batch_first=True, bidirectional=is_bidirectional, device=device))
            prev_size = hidden_size * (1 if not is_bidirectional else 2)
            
        self.fc_layers = nn.ModuleList()
        prev_size = lstm_hidden_sizes[-1] * (1 if not is_bidirectional else 2)
        
        for fc_size in fc_sizes[:-1]:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))
            prev_size = fc_size
            
        self.fc_layers.append(nn.Linear(prev_size, 1))
        # self.sigmoid = nn.Sigmoid()

        self.to(device)
        
    def forward(self, x):
        lstm_out = x
        for lstm_layer in self.lstm_layers:
            lstm_out, _ = lstm_layer(lstm_out)
        
        out = lstm_out[:, -1, :]
        
        for layer in self.fc_layers:
            out = layer(out)
            
        # return self.sigmoid(out)
        return out
    
class LSTMNetworkWithMask(LSTMNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_mask = FeatureMaskLayer(kwargs['input_size'], kwargs['device'])
        self.device = kwargs['device']

    # 0 - disabled, 1 - enabled 
    def set_mask(self, disabled_features: list[int]):
        self.clear_mask()
        self.feature_mask.mask[disabled_features] = 0

    def clear_mask(self):
        self.feature_mask.mask = torch.ones_like(self.feature_mask.mask, device=self.device)

    def forward(self, x):
        x = self.feature_mask(x)
        return super().forward(x)

class MultiSequenceLSTMNetwork(nn.Module):
    def __init__(self, 
        input_size: int,
        sequence_lengths: list[int],
        lstm_hidden_sizes: list[int],
        fc_sizes: list[int],
        dropout_rate: float,
        lstm_layers: int,
        is_bidirectional: bool,
        device: torch.device
    ):
        super().__init__()
        self.sequence_lengths = sorted(sequence_lengths, reverse=True)  # Sort descending for slicing
        self.input_size = input_size
        self.device = device
        
        # Create LSTM branch for each sequence length
        self.lstm_branches = nn.ModuleList()
        
        for _ in sequence_lengths:
            branch = nn.ModuleList()
            prev_size = input_size
            
            for hidden_size in lstm_hidden_sizes:
                branch.append(
                    nn.LSTM(
                        prev_size, 
                        hidden_size, 
                        num_layers=lstm_layers, 
                        dropout=dropout_rate, 
                        batch_first=True, 
                        bidirectional=is_bidirectional,
                        device=device
                    )
                )
                prev_size = hidden_size * (2 if is_bidirectional else 1)
            
            self.lstm_branches.append(branch)
            
        # Calculate total size for FC input
        total_lstm_output = len(sequence_lengths) * lstm_hidden_sizes[-1] * (2 if is_bidirectional else 1)
        
        # Create FC layers
        self.fc_layers = nn.ModuleList()
        prev_size = total_lstm_output
        
        for fc_size in fc_sizes[:-1]:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))
            prev_size = fc_size
            
        self.fc_layers.append(nn.Linear(prev_size, fc_sizes[-1]))
        
        self.to(device)
        
    def forward(self, x):
        # Process input through each LSTM branch with different sequence lengths
        lstm_outputs = []
        
        for seq_len, lstm_branch in zip(self.sequence_lengths, self.lstm_branches):
            # Take last seq_len elements from sequence
            x_slice = x[:, -seq_len:, :]
            lstm_out = x_slice
            
            for lstm_layer in lstm_branch:
                lstm_out, _ = lstm_layer(lstm_out)
            
            # Take the last output from sequence
            lstm_outputs.append(lstm_out[:, -1, :])
        
        # Concatenate all LSTM outputs
        combined = torch.cat(lstm_outputs, dim=1)
        
        # Process through FC layers
        out = combined
        for layer in self.fc_layers:
            out = layer(out)
            
        return out

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.to(device)

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        
        # Используем torch.cat с dim=1 для оптимизации памяти
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        
        # Используем torch.chunk вместо split для оптимизации памяти
        i, f, g, o = conv_output.chunk(4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class BidirectionalConvLSTMLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.forward_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, device)
        self.backward_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, device)
        self.to(device)

    def forward(self, x, spatial_dims):
        batch_size, seq_length = x.size(0), x.size(1)
        h_forward, c_forward = self._init_hidden(batch_size, spatial_dims)
        h_backward, c_backward = self._init_hidden(batch_size, spatial_dims)
        
        # Предварительно выделяем память для выходных тензоров
        output = torch.empty(batch_size, seq_length, self.hidden_channels * 2, 
                           spatial_dims[0], spatial_dims[1], 
                           device=x.device, dtype=x.dtype)
        
        # Прямой проход
        h_state = h_forward
        c_state = c_forward
        for t in range(seq_length):
            h_state, c_state = self.forward_cell(x[:, t], (h_state, c_state))
            output[:, t, :self.hidden_channels] = h_state
        
        # Обратный проход
        h_state = h_backward
        c_state = c_backward
        for t in range(seq_length - 1, -1, -1):
            h_state, c_state = self.backward_cell(x[:, t], (h_state, c_state))
            output[:, t, self.hidden_channels:] = h_state
        
        return output

    def _init_hidden(self, batch_size, spatial_dims):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_channels, spatial_dims[0], spatial_dims[1], 
                       device=device, dtype=torch.float32),
            torch.zeros(batch_size, self.hidden_channels, spatial_dims[0], spatial_dims[1], 
                       device=device, dtype=torch.float32)
        )

class ConvLSTMNetwork(nn.Module):
    def __init__(self, 
        input_size: int,
        lstm_hidden_sizes: list[int],
        fc_sizes: list[int],
        dropout_rate: float,
        lstm_layers: int,
        is_bidirectional: bool,
        device: torch.device
    ):
        super().__init__()
        
        self.reshape_height = 13
        self.reshape_width = input_size // self.reshape_height
        self.is_bidirectional = is_bidirectional
        
        # Инициализация слоев
        self.convlstm_layers = nn.ModuleList()
        prev_channels = 1
        
        for hidden_size in lstm_hidden_sizes:
            for _ in range(lstm_layers):
                if is_bidirectional:
                    layer = BidirectionalConvLSTMLayer(
                        input_channels=prev_channels,
                        hidden_channels=hidden_size // 2,
                        kernel_size=3,
                        device=device
                    )
                    prev_channels = hidden_size
                else:
                    layer = ConvLSTMCell(
                        input_channels=prev_channels,
                        hidden_channels=hidden_size,
                        kernel_size=3,
                        device=device
                    )
                    prev_channels = hidden_size
                self.convlstm_layers.append(layer)
        
        # Вычисление размера входа для FC слоев
        if is_bidirectional:
            fc_input_size = lstm_hidden_sizes[-1] * self.reshape_height * self.reshape_width
        else:
            fc_input_size = lstm_hidden_sizes[-1] * self.reshape_height * self.reshape_width

        # FC слои
        self.fc_layers = nn.ModuleList()
        prev_size = fc_input_size
        
        for fc_size in fc_sizes[:-1]:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))
            prev_size = fc_size
            
        self.fc_layers.append(nn.Linear(prev_size, fc_sizes[-1]))
        # self.sigmoid = nn.Sigmoid()
        
        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Преобразование входных данных
        current_input = x.view(batch_size, seq_length, 1, self.reshape_height, self.reshape_width)
        
        if self.is_bidirectional:
            for layer in self.convlstm_layers:
                current_input = layer(current_input, (self.reshape_height, self.reshape_width))
            out = current_input[:, -1]
        else:
            for layer in self.convlstm_layers:
                h, c = self._init_hidden(batch_size, layer.hidden_channels)
                for t in range(seq_length):
                    h, c = layer(current_input[:, t], (h, c))
                out = h
        
        # Преобразование для FC слоев
        out = out.view(batch_size, -1)
        
        # Проход через FC слои
        for layer in self.fc_layers:
            out = layer(out)
            
        # return self.sigmoid(out)
        return out

    def _init_hidden(self, batch_size, hidden_channels):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, hidden_channels, self.reshape_height, self.reshape_width, 
                       device=device, dtype=torch.float32),
            torch.zeros(batch_size, hidden_channels, self.reshape_height, self.reshape_width, 
                       device=device, dtype=torch.float32)
        )