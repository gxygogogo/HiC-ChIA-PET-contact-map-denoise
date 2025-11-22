import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import gzip
import sys
import torch.optim as optim

conv2d1_filters_numbers = 8
conv2d1_filters_size = 5
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 3


class conv_autoencoder3(nn.Module):
    def __init__(self):
        super(conv_autoencoder3, self).__init__()

        self.SM_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers,conv2d1_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers,conv2d2_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size),
            nn.ReLU()
        )

        self.SM_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size),
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size), 
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size),  
            nn.ReLU()
        )

        self.SA1_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers,conv2d1_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers,conv2d2_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size),
            nn.ReLU()
        )

        self.SA1_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size),
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size), 
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size),  
            nn.ReLU()
        )
        
        self.SA2_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers,conv2d1_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers,conv2d2_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size),
            nn.ReLU()
        )

        self.SA2_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size),
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size), 
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size),  
            nn.ReLU()
        )

    def forward(self, x1, x2, x3):
        SM_encode = self.SM_encoder(x1)
        x1 = self.SM_decoder(SM_encode)
        SA1_encode = self.SM_encoder(x2)
        x2 = self.SM_decoder(SA1_encode)
        SA2_encode = self.SM_encoder(x3)
        x3 = self.SM_decoder(SA2_encode)
        return  x1, x2, x3, SM_encode, SA1_encode, SA2_encode
        

class conv_autoencoder2(nn.Module):
    def __init__(self):
        super(conv_autoencoder2, self).__init__()

        self.SM_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers,conv2d1_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers,conv2d2_filters_size),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size),
            nn.ReLU()
        )

        self.SM_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size),
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size), 
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size),  
            nn.ReLU()
        )

        self.SA1_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers,conv2d1_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers,conv2d2_filters_size),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size),
            nn.ReLU()
        )

        self.SA1_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size),
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size), 
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size),  
            nn.ReLU()
        )
        
        self.SA2_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers,conv2d1_filters_size),
            nn.ReLU(),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers,conv2d2_filters_size),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size),
            nn.ReLU()
        )

        self.SA2_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size),
            nn.ReLU(),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size), 
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size),  
            nn.ReLU()
        )

    def forward(self, x1, x2, x3):
        SM_encode = self.SM_encoder(x1)
        x1 = self.SM_decoder(SM_encode)
        SA1_encode = self.SM_encoder(x2)
        x2 = self.SM_decoder(SA1_encode)
        SA2_encode = self.SM_encoder(x3)
        x3 = self.SM_decoder(SA2_encode)
        return  x1, x2, x3, SM_encode, SA1_encode, SA2_encode
    




class conv_autoencoder_pro(nn.Module):
    def __init__(self):
        super(conv_autoencoder_pro, self).__init__()

        # SM Encoder
        self.SM_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SM Decoder
        self.SM_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2)
        )

        # SA1 Encoder
        self.SA1_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SA1 Decoder
        self.SA1_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2)
        )
        
        # SA2 Encoder
        self.SA2_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SA2 Decoder
        self.SA2_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2)
        )
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x1, x2, x3):
        # SM Encoder
        SM_encode = self.SM_encoder(x1)
        SM_encode = self.dropout(SM_encode)
        
        # SM Decoder with residual connection
        x1_decode = self.SM_decoder(SM_encode)
        x1_decode = self.dropout(x1_decode)
        x1_decode += x1  # Adding residual connection
        x1_decode = nn.ReLU(inplace=True)(x1_decode)  # Applying ReLU after adding
        
        # SA1 Encoder
        SA1_encode = self.SA1_encoder(x2)
        SA1_encode = self.dropout(SA1_encode)
        
        # SA1 Decoder with residual connection
        x2_decode = self.SA1_decoder(SA1_encode)
        x2_decode = self.dropout(x2_decode)
        x2_decode += x2  # Adding residual connection
        x2_decode = nn.ReLU(inplace=True)(x2_decode)  # Applying ReLU after adding

        # SA2 Encoder
        SA2_encode = self.SA2_encoder(x3)
        SA2_encode = self.dropout(SA2_encode)
        
        # SA2 Decoder with residual connection
        x3_decode = self.SA2_decoder(SA2_encode)
        x3_decode = self.dropout(x3_decode)
        x3_decode += x3  # Adding residual connection
        x3_decode = nn.ReLU(inplace=True)(x3_decode)  # Applying ReLU after adding

        return x1_decode, x2_decode, x3_decode, SM_encode, SA1_encode, SA2_encode


# v16

class conv_autoencoder_res(nn.Module):
    def __init__(self):
        super(conv_autoencoder_res, self).__init__()

        # SM Encoder
        self.SM_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SM Decoder
        self.SM_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2)
        )

        # SA1 Encoder
        self.SA1_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SA1 Decoder
        self.SA1_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2)
        )
        
        # SA2 Encoder
        self.SA2_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 2, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SA2 Decoder
        self.SA2_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2)
        )
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x1, x2, x3):
        # SM Encoder
        SM_encode = self.SM_encoder(x1)
        SM_encode = self.dropout(SM_encode)
        
        # SM Decoder with residual connection
        x1_decode = self.SM_decoder(SM_encode)
        x1_decode = self.dropout(x1_decode)
        x1_decode += x1  # Adding residual connection
        x1_decode = nn.ReLU(inplace=True)(x1_decode)  # Applying ReLU after adding
        
        # SA1 Encoder
        SA1_encode = self.SA1_encoder(x2)
        SA1_encode = self.dropout(SA1_encode)
        
        # SA1 Decoder with residual connection
        x2_decode = self.SA1_decoder(SA1_encode)
        x2_decode = self.dropout(x2_decode)
        x2_decode += x2  # Adding residual connection
        x2_decode = nn.ReLU(inplace=True)(x2_decode)  # Applying ReLU after adding

        # SA2 Encoder
        SA2_encode = self.SA2_encoder(x3)
        SA2_encode = self.dropout(SA2_encode)
        
        # SA2 Decoder with residual connection
        x3_decode = self.SA2_decoder(SA2_encode)
        x3_decode = self.dropout(x3_decode)
        x3_decode += x3  # Adding residual connection
        x3_decode = nn.ReLU(inplace=True)(x3_decode)  # Applying ReLU after adding

        return x1_decode, x2_decode, x3_decode, SM_encode, SA1_encode, SA2_encode

# v17
class conv_autoencoder_res2(nn.Module):
    def __init__(self):
        super(conv_autoencoder_res2, self).__init__()

        # SM Encoder
        self.SM_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 3, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SM Decoder
        self.SM_decoder = nn.Sequential(
            nn.ConvTranspose2d(3, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True)
        )

        # SA1 Encoder
        self.SA1_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 3, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SA1 Decoder
        self.SA1_decoder = nn.Sequential(
            nn.ConvTranspose2d(3, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # SA2 Encoder
        self.SA2_encoder = nn.Sequential(
            nn.Conv2d(3, conv2d1_filters_numbers, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2d2_filters_numbers, 3, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True)
        )

        # SA2 Decoder
        self.SA2_decoder = nn.Sequential(
            nn.ConvTranspose2d(3, conv2d2_filters_numbers, conv2d3_filters_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d2_filters_numbers, conv2d1_filters_numbers, conv2d2_filters_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv2d1_filters_numbers, 3, conv2d1_filters_size, padding=2),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x1, x2, x3):
        # SM Encoder
        x1_encode = self.SM_encoder(x1)

        # Adding residual connection after SM Encoder
        x1_encode = x1 + x1_encode
        x1_encode = nn.ReLU(inplace=True)(x1_encode)
        x1_encode = self.dropout(x1_encode)
        
        # SM Decoder
        x1_decode = self.SM_decoder(x1_encode)
        
        # Adding residual connection after SM Decoder
        x1_decode = x1 + x1_decode
        x1_decode = nn.ReLU(inplace=True)(x1_decode)
        x1_decode = self.dropout(x1_decode)

        # SA1 Encoder
        x2_encode = self.SA1_encoder(x2)

        # Adding residual connection after SM Encoder
        x2_encode = x2 + x2_encode
        x2_encode = nn.ReLU(inplace=True)(x2_encode)
        x2_encode = self.dropout(x2_encode)
        
        # SA1 Decoder
        x2_decode = self.SA1_decoder(x2_encode)
        
        # Adding residual connection after SA1 Decoder
        x2_decode = x2 + x2_decode
        x2_decode = nn.ReLU(inplace=True)(x2_decode)
        x2_decode = self.dropout(x2_decode)

        # SA2 Encoder
        x3_encode = self.SA2_encoder(x3)

        # Adding residual connection after SM Encoder
        x3_encode = x3 + x3_encode
        x3_encode = nn.ReLU(inplace=True)(x3_encode)
        x3_encode = self.dropout(x3_encode)
        
        # SA1 Decoder
        x3_decode = self.SA2_decoder(x3_encode)
        
        # Adding residual connection after SA1 Decoder
        x3_decode = x3 + x3_decode
        x3_decode = nn.ReLU(inplace=True)(x3_decode)
        x3_decode = self.dropout(x3_decode)

        return x1_decode, x2_decode, x3_decode, x1_decode, x2_decode, x3_decode
