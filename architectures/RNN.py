import torch
import torch.nn as nn

# Defining LSTM RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()     # inheriting from existing RNN class
        self.num_layers = num_layers    # number of input layers
        self.hidden_size = hidden_size  # number of hidden players

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # creating LSTM layer
        self.fc = nn.Linear(hidden_size, num_classes)                               # creating linear output layer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # x -> (batch_size, seq_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out -> (batch_size, seq_size, input_size) = (N, 50, 512)
        out = out[:, -1, :]
        # out -> (N, 512)
        out = self.fc(out)


        return torch.sigmoid(out) # returning one forward step of the NN