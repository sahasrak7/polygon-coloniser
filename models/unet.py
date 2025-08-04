import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottom
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Color embedding (8 colors from JSON)
        self.color_embed = nn.Embedding(8, 64)
    
    def forward(self, x, color_idx):
        skip_connections = []
        
        # Embed color
        color_embedding = self.color_embed(color_idx).unsqueeze(-1).unsqueeze(-1)
        color_embedding = color_embedding.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, color_embedding], dim=1)
        
        # Down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        # Up
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        
        return self.final_conv(x)