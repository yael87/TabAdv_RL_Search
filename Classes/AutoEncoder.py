
import torch.nn as nn
import torch
    
class AutoEncoder(nn.Module):
    def __init__(self, in_deg, encoding_dim):
        super().__init__()
        #self.encoder_hidden_layer = nn.Linear(in_features=in_deg, out_features=2048)
        #self.encoder_output_layer = nn.Linear(in_features=2048, out_features=encoding_dim)
        #self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        #self.decoder_output_layer = nn.Linear(in_features=128, out_features=in_deg)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_deg, encoding_dim),
            torch.nn.ReLU(),
            #torch.nn.Linear(128, 64),
            #torch.nn.ReLU(),
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, in_deg),
            #torch.nn.ReLU(),
            #torch.nn.Linear(128, 28 * 28),
            #torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded
    