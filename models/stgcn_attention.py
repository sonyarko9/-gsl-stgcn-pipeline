import torch
import torch.nn as nn


class STGCNAttention(nn.Module):
    """
    ST-GCN with a lightweight Multi-Head Self-Attention layer appended
    after the temporal output, before the classification head.
    This constitutes the GCN-Attention hybrid described in the proposal.
    """

    def __init__(self, in_channels, num_classes, A, num_heads=4, d_model=256):
        super(STGCNAttention, self).__init__()

        # --- ST-GCN layers (spatial-temporal graph convolutions) ---
        self.stgcn_layers = nn.ModuleList([
            STGCNLayer(in_channels, 64, A),
            STGCNLayer(64, 64, A),
            STGCNLayer(64, 64, A),
            STGCNLayer(64, 128, A),
            STGCNLayer(128, 128, A),
            STGCNLayer(128, 128, A),
            STGCNLayer(128, 256, A),
            STGCNLayer(256, 256, A),
            STGCNLayer(256, 256, A),
            STGCNLayer(256, d_model, A),  # final layer outputs d_model=256
        ])

        # --- THE UPGRADE: Multi-Head Self-Attention layer ---
        # This is the 4-line addition the supervisor requires
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,   # must match ST-GCN final output size (256)
            num_heads=num_heads, # 4 attention heads as specified
            batch_first=True     # input shape: (batch, sequence, features)
        )

        # --- Classification head ---
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape coming in: (batch, channels, frames, joints)

        # Step 1: Pass through all 10 ST-GCN layers
        for layer in self.stgcn_layers:
            x = layer(x)

        # Step 2: Global average pool over joints
        # Shape becomes: (batch, d_model, frames)
        x = x.mean(dim=-1)

        # Step 3: Reshape for attention
        # Attention expects: (batch, sequence_length, features)
        # sequence_length = frames, features = d_model
        x = x.permute(0, 2, 1)

        # Step 4: THE UPGRADE — apply self-attention
        # Query, Key, Value are all x (self-attention)
        # This lets every frame attend to every other frame
        x, _ = self.attn(x, x, x)

        # Step 5: Pool over the time dimension
        x = x.mean(dim=1)

        # Step 6: Dropout + classification head
        x = self.dropout(x)
        x = self.fc(x)

        return x


class STGCNLayer(nn.Module):
    """Single ST-GCN layer: spatial graph conv + temporal conv."""

    def __init__(self, in_channels, out_channels, A):
        super(STGCNLayer, self).__init__()

        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32),
                              requires_grad=False)

        # Spatial graph convolution
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Temporal convolution (across frames)
        self.tcn = nn.Conv2d(out_channels, out_channels,
                             kernel_size=(9, 1),
                             padding=(4, 0))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Residual connection if channel sizes differ
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)

        # Spatial: multiply features by adjacency matrix
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.gcn(x)

        # Temporal: convolve across frames
        x = self.tcn(x)
        x = self.bn(x)

        return self.relu(x + res)