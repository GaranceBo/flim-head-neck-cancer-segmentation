############################################
# Pytorch pretrained U-net model (Resnet34 backbone pretrained on ImageNet dataset)
############################################

def make_pretrained_unet(input_channels=3, encoder_name='resnet34', encoder_weights='imagenet'):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1,          # binary segmentation -> single-channel logits
        activation=None     # returns raw logits (we use BCEWithLogitsLoss)
    )
    return model

############################################
# Simple ViT Transformer model for binary pixel classification 
# Based on: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
############################################

# 1. Patch Embedding for ViT
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embedding_dim, num_channels):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.proj = nn.Linear(patch_size * patch_size * num_channels, embedding_dim)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size, channels, height, width = x.shape
        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: (batch, channels, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (batch, num_patches, channels, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, patches.shape[1], -1)  # flatten patches
        embeddings = self.proj(patches)  # (batch, num_patches, embedding_dim)
        return embeddings

# 2. Transformer Encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, embedding_dim)
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        x_norm2 = self.norm2(x)
        x = x + self.mlp(x_norm2)
        return x

# 3. ViT segmentation model
class ViTSegmentation(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        image_size = input_shape[0]
        num_channels = input_shape[2]
        patch_size = 8
        self.num_patches = (image_size // patch_size) ** 2
        embedding_dim = 256
        num_heads = 4
        transformer_layers = 4
        mlp_dim = 128

        self.patch_embed = PatchEmbedding(patch_size, embedding_dim, num_channels)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim))
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embedding_dim, num_heads, mlp_dim) for _ in range(transformer_layers)]
        )
        self.conv_transpose = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = nn.LayerNorm(embedding_dim)
        self.gelu = nn.GELU()
        self.final_conv = nn.Conv2d(embedding_dim, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.patch_embed(x)  # (batch, num_patches, embedding_dim)
        x = x + self.pos_emb
        for encoder in self.transformer_encoders:
            x = encoder(x)
        h = w = int(self.num_patches ** 0.5)
        x = x.permute(0, 2, 1).contiguous().view(x.size(0), -1, h, w)  # (batch, embedding_dim, h, w)
        x = self.conv_transpose(x)  # upsample to original size
        # LayerNorm expects (batch, seq_len, embedding_dim), so permute and flatten spatial dims
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        x = self.norm(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.gelu(x)
        x = self.final_conv(x)
        return x

############################################
# Simple U-net with Dropout (0.3) after every pooling and upsampling layer.
############################################

class SimpleUNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.drop4 = nn.Dropout(0.3)

        self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.drop5 = nn.Dropout(0.3)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.drop1(self.pool1(c1))

        c2 = self.conv2(p1)
        p2 = self.drop2(self.pool2(c2))

        c3 = self.conv3(p2)

        u4 = self.up4(c3)
        u4 = torch.cat([u4, c2], dim=1)
        c4 = self.conv4(u4)
        c4 = self.drop4(c4)

        u5 = self.up5(c4)
        u5 = torch.cat([u5, c1], dim=1)
        c5 = self.conv5(u5)
        c5 = self.drop5(c5)

        out = self.final(c5)
        return out
    
############################################
# Hybrid U-net CBAM model
# Based on: A Spatial Distribution Extraction Method for Winter Wheat Based on Improved U-Net, Jiahao Liu
############################################

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
        
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_att = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x

# ASPP module
class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_3x3_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_3x3_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_3x3_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[2:]
        image_pool = self.image_pool(x)
        image_pool = F.interpolate(image_pool, size=size, mode='bilinear', align_corners=False)
        x1 = self.conv_1x1(x)
        x2 = self.conv_3x3_6(x)
        x3 = self.conv_3x3_12(x)
        x4 = self.conv_3x3_18(x)
        x = torch.cat([x1, x2, x3, x4, image_pool], dim=1)
        x = self.conv_out(x)
        return x

# RAUnet model
class RAUnet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super().__init__()
        self.e1 = ResidualBlock(input_channels, 32)
        self.cbam1 = CBAMBlock(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.3)

        self.e2 = ResidualBlock(32, 64)
        self.cbam2 = CBAMBlock(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3)

        self.e3 = ResidualBlock(64, 128)
        self.cbam3 = CBAMBlock(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.3)

        self.e4 = ResidualBlock(128, 256)
        self.cbam4 = CBAMBlock(256)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(0.3)

        self.bridge = ResidualBlock(256, 512)
        self.aspp = ASPPModule(512, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d4_res = ResidualBlock(512, 256)
        self.dropd4 = nn.Dropout(0.3)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3_res = ResidualBlock(256, 128)
        self.dropd3 = nn.Dropout(0.3)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2_res = ResidualBlock(128, 64)
        self.dropd2 = nn.Dropout(0.3)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1_res = ResidualBlock(64, 32)
        self.dropd1 = nn.Dropout(0.3)

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e1_cbam = self.cbam1(e1)
        p1 = self.drop1(self.pool1(e1))

        e2 = self.e2(p1)
        e2_cbam = self.cbam2(e2)
        p2 = self.drop2(self.pool2(e2))

        e3 = self.e3(p2)
        e3_cbam = self.cbam3(e3)
        p3 = self.drop3(self.pool3(e3))

        e4 = self.e4(p3)
        e4_cbam = self.cbam4(e4)
        p4 = self.drop4(self.pool4(e4))

        b = self.bridge(p4)
        b = self.aspp(b)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4_cbam], dim=1)
        d4 = self.dropd4(self.d4_res(d4))

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3_cbam], dim=1)
        d3 = self.dropd3(self.d3_res(d3))

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2_cbam], dim=1)
        d2 = self.dropd2(self.d2_res(d2))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1_cbam], dim=1)
        d1 = self.dropd1(self.d1_res(d1))

        out = self.final(d1)
        return out
    
############################################
# 34-layer ResNet Model (pretrained) for binary segmentation
############################################

class ResNet34Segmentation32(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load ResNet34 backbone
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        # Use only early-to-mid layers (avoid over-downsampling)
        self.encoder = nn.Sequential(
            resnet.conv1,     # 64, stride=2
            resnet.bn1,
            resnet.relu,
            # Skip maxpool to preserve spatial size!
            resnet.layer1,    # stride=1
            resnet.layer2,    # stride=2
            resnet.layer3,    # stride=2
            # omit layer4 to avoid collapsing small inputs
        )

        # Decoder (upsampling back to 32x32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1) #,
            #nn.Sigmoid(),  # pixel-wise probabilities 0–1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

############################################
# Vgg Model
############################################

class VGGSegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder (VGG-like)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (upsampling path)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)  # (B,64,32,32)
        x2 = self.enc2(self.pool1(x1))  # (B,128,16,16)
        x3 = self.enc3(self.pool2(x2))  # (B,256,8,8)

        d2 = self.up2(x3)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

############################################
# FastFCN Model
############################################

# Pyramid Pooling Module (PPM)
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 4)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
                nn.ReLU(inplace=True)
            )
            for ps in pool_sizes
        ])

    def forward(self, x):
        h, w = x.shape[2:]
        pyramids = [x]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(out)
        return torch.cat(pyramids, dim=1)

# FastFCN-style model for 32x32 images
class FastFCN32(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        # Encoder: Use ResNet layers up to layer3
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,  # stop before layer4 to keep enough spatial resolution
        )

        # Context module (Pyramid Pooling)
        self.ppm = PyramidPooling(256, pool_sizes=(1, 2, 4))

        # Decoder / upsampling head
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 3 * (256 // 3), 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),  # logits output
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.ppm(x)
        x = self.decoder(x)
        return x

############################################
# FastFCN variant 
############################################

# --- JPU module (Joint Pyramid Upsampling) ---
class JPUModule(nn.Module):
    def __init__(self, in_channels, width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels[0], width, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels[1], width, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels[2], width, 3, padding=1)

        # Parallel dilated convolutions
        self.dilation_convs = nn.ModuleList([
            nn.Conv2d(3 * width, width, 3, padding=d, dilation=d) for d in [1, 2, 4, 8]
        ])

    def forward(self, feats):
        feat1 = self.conv1(feats[0])
        feat2 = F.interpolate(self.conv2(feats[1]), size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat3 = F.interpolate(self.conv3(feats[2]), size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([feat1, feat2, feat3], dim=1)

        outs = [conv(feat) for conv in self.dilation_convs]
        return torch.cat(outs, dim=1)  # concat dilated features

# --- Encoder block ---
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

# --- Lightweight FastFCN ---
class MiniFastFCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)

        self.jpu = JPUModule([32, 64, 128], width=64)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64 * 4, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        f1 = self.enc1(x)  # 32x32
        f2 = self.enc2(self.pool1(f1))  # 16x16
        f3 = self.enc3(self.pool2(f2))  # 8x8
        jpu_out = self.jpu([f1, f2, f3])
        out = self.dec(jpu_out)
        # Ensure output matches target size
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out

############################################
# Light RAUnet
############################################

# ===== Residual Block =====
class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

# ===== CBAM Block (lightweight) =====
class CBAMBlock2(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1)
        maxp = F.adaptive_max_pool2d(x, 1)
        channel_att = self.sigmoid(self.mlp(avg) + self.mlp(maxp))
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

# ===== Lightweight ASPP =====
class ASPPModule2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1),
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4),
        ])
        self.project = nn.Conv2d(out_channels * 3, out_channels, 1)

    def forward(self, x):
        feats = [F.relu(conv(x)) for conv in self.convs]
        return self.project(torch.cat(feats, dim=1))

# ===== RAUNet-Lite =====
class RAUNetLite(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(ResidualBlock2(in_channels, 32), CBAMBlock2(32))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(ResidualBlock2(32, 64), CBAMBlock2(64))
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = nn.Sequential(ResidualBlock2(64, 128), ASPPModule2(128, 128))

        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock2(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResidualBlock2(32 + 32, 32)

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)          # (B,32,32,32)
        e2 = self.enc2(self.pool1(e1))  # (B,64,16,16)
        b  = self.bridge(self.pool2(e2)) # (B,128,8,8)

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out
        
############################################
############################################ ###############################################
# NEW MODELS
###############################################
# ATTENTION U-NET
###############################################
# -------------------------
# Helpers (Conv blocks)
# -------------------------
class ConvBlock2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # ConvTranspose2d for upsampling (like your UNet)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x):
        return self.up(x)

# -------------------------
# Attention Gate (Oktay et al.)
# -------------------------
class AttentionGate(nn.Module):
    """
    Attention gate that takes skip connection (x) and gating signal (g), 
    returns x * attention_coeff where attention_coeff in [0,1].
    """
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        # Theta_x: reduces skip channels
        self.theta_x = nn.Conv2d(in_ch, inter_ch, kernel_size=2, stride=2, bias=False)
        # Phi_g: reduces gating channels
        self.phi_g = nn.Conv2d(gating_ch, inter_ch, kernel_size=1, bias=True)
        # f: combine
        self.f = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # final upsample to match x spatial
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, g):
        """
        x: skip connection (B, in_ch, H, W)
        g: gating signal (B, gating_ch, Hg, Wg) - typically smaller spatial size
        """
        # reduce spatial size of x to match coarse g
        theta_x = self.theta_x(x)             # (B, inter_ch, Hg, Wg)
        phi_g = self.phi_g(g)                 # (B, inter_ch, Hg, Wg)

        # Resize phi_g to theta_x spatial size
        phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode='bilinear', align_corners=False)
        
        add = theta_x + phi_g
        
        psi = self.f(add)                     # (B,1,Hg,Wg) in [0,1]
        psi_up = self.up_sample(psi)          # (B,1,H,W)
        # broadcast multiply
        return x * psi_up

# -------------------------
# Attention U-Net (look-alike)
# -------------------------
class AttentionUNet(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        """
        Look-alike Attention U-Net (Oktay et al.)
        - input_channels: number of input channels (3)
        - base_filters: filters for first level (default 64 like your UNet)
        """
        super().__init__()
        f = base_filters
        # Encoder
        self.conv1 = ConvBlock2(input_channels, f)         # -> f
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = ConvBlock2(f, f*2)                    # -> 2f
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = ConvBlock2(f*2, f*4)                  # -> 4f
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = ConvBlock2(f*4, f*8)                  # -> 8f
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(0.3)

        # Bottleneck
        self.conv5 = ConvBlock2(f*8, f*16)                 # -> 16f

        # Decoder (upsampling + attention gates)
        self.up6 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.att6 = AttentionGate(in_ch=f*8, gating_ch=f*8, inter_ch=f*4)
        self.conv6 = ConvBlock2(f*16, f*8)
        self.drop6 = nn.Dropout(0.3)

        self.up7 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.att7 = AttentionGate(in_ch=f*4, gating_ch=f*4, inter_ch=f*2)
        self.conv7 = ConvBlock2(f*8, f*4)
        self.drop7 = nn.Dropout(0.3)

        self.up8 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.att8 = AttentionGate(in_ch=f*2, gating_ch=f*2, inter_ch=f)
        self.conv8 = ConvBlock2(f*4, f*2)
        self.drop8 = nn.Dropout(0.3)

        self.up9 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.att9 = AttentionGate(in_ch=f, gating_ch=f, inter_ch=max(f//2, 8))
        self.conv9 = ConvBlock2(f*2, f)
        self.drop9 = nn.Dropout(0.3)

        self.final = nn.Conv2d(f, 1, kernel_size=1)

    def _resize_to(self, src, target):
        """Utility: resize src to target's spatial size if needed."""
        if src.shape[2:] == target.shape[2:]:
            return src
        return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)            # (B, f, H, W)
        p1 = self.drop1(self.pool1(c1))

        c2 = self.conv2(p1)           # (B, 2f, H/2, W/2)
        p2 = self.drop2(self.pool2(c2))

        c3 = self.conv3(p2)           # (B, 4f, H/4, W/4)
        p3 = self.drop3(self.pool3(c3))

        c4 = self.conv4(p3)           # (B, 8f, H/8, W/8)
        p4 = self.drop4(self.pool4(c4))

        # Bottleneck
        c5 = self.conv5(p4)           # (B, 16f, H/16, W/16)

        # Decoder level 6 (connect c4)
        u6 = self.up6(c5)             # (B, 8f, H/8, W/8)
        # align if shapes differ
        u6 = self._resize_to(u6, c4)
        # attention gate on skip connection c4 using gating signal u6
        g6 = u6
        c4_att = self.att6(c4, g6)    # (B, 8f, H/8, W/8)
        u6 = torch.cat([u6, c4_att], dim=1)
        c6 = self.conv6(u6)
        c6 = self.drop6(c6)

        # Decoder level 7 (connect c3)
        u7 = self.up7(c6)             # (B, 4f, H/4, W/4)
        u7 = self._resize_to(u7, c3)
        g7 = u7
        c3_att = self.att7(c3, g7)
        u7 = torch.cat([u7, c3_att], dim=1)
        c7 = self.conv7(u7)
        c7 = self.drop7(c7)

        # Decoder level 8 (connect c2)
        u8 = self.up8(c7)             # (B, 2f, H/2, W/2)
        u8 = self._resize_to(u8, c2)
        g8 = u8
        c2_att = self.att8(c2, g8)
        u8 = torch.cat([u8, c2_att], dim=1)
        c8 = self.conv8(u8)
        c8 = self.drop8(c8)

        # Decoder level 9 (connect c1)
        u9 = self.up9(c8)             # (B, f, H, W)
        u9 = self._resize_to(u9, c1)
        g9 = u9
        c1_att = self.att9(c1, g9)
        u9 = torch.cat([u9, c1_att], dim=1)
        c9 = self.conv9(u9)
        c9 = self.drop9(c9)

        out = self.final(c9)          # (B, 1, H, W)
        return out

###############################################
# U-net ++ 
###############################################

# ---- Basic blocks ----
class ConvBlock3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    def forward(self, x): return self.up(x)


# ---- UNet++ ----
class UNetPP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        filters = [base, base*2, base*4, base*8, base*16]

        self.conv00 = ConvBlock3(in_ch, filters[0])
        self.conv10 = ConvBlock3(filters[0], filters[1])
        self.conv20 = ConvBlock3(filters[1], filters[2])
        self.conv30 = ConvBlock3(filters[2], filters[3])
        self.conv40 = ConvBlock3(filters[3], filters[4])

        self.up01 = Up(filters[1], filters[0])
        self.up11 = Up(filters[2], filters[1])
        self.up21 = Up(filters[3], filters[2])
        self.up31 = Up(filters[4], filters[3])

        self.conv01 = ConvBlock3(filters[0] + filters[0], filters[0])
        self.conv11 = ConvBlock3(filters[1] + filters[1], filters[1])
        self.conv21 = ConvBlock3(filters[2] + filters[2], filters[2])
        self.conv31 = ConvBlock3(filters[3] + filters[3], filters[3])

        self.up02 = Up(filters[1], filters[0])
        self.up12 = Up(filters[2], filters[1])
        self.up22 = Up(filters[3], filters[2])
        self.cut = lambda a,b: a[:, :, :b.size(2), :b.size(3)]

        self.conv02 = ConvBlock3(filters[0]*3, filters[0])
        self.conv12 = ConvBlock3(filters[1]*3, filters[1])
        self.conv22 = ConvBlock3(filters[2]*3, filters[2])

        self.up03 = Up(filters[1], filters[0])
        self.conv03 = ConvBlock3(filters[0]*4, filters[0])

        self.up04 = Up(filters[1], filters[0])
        self.conv04 = ConvBlock3(filters[0]*5, filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(F.max_pool2d(x00, 2))
        x20 = self.conv20(F.max_pool2d(x10, 2))
        x30 = self.conv30(F.max_pool2d(x20, 2))
        x40 = self.conv40(F.max_pool2d(x30, 2))

        x01 = self.conv01(torch.cat([x00, self.up01(x10)], 1))
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], 1))
        x21 = self.conv21(torch.cat([x20, self.up21(x30)], 1))
        x31 = self.conv31(torch.cat([x30, self.up31(x40)], 1))

        x02 = self.conv02(torch.cat([x00, self.cut(self.up02(x11), x00), x01], 1))
        x12 = self.conv12(torch.cat([x10, self.cut(self.up12(x21), x10), x11], 1))
        x22 = self.conv22(torch.cat([x20, self.cut(self.up22(x31), x20), x21], 1))

        x03 = self.conv03(torch.cat([x00, self.up03(x12), x01, x02], 1))

        x04 = self.conv04(torch.cat([x00, self.cut(self.up04(x12), x00), x01, x02, x03], 1))

        #return torch.sigmoid(self.final(x04))
        return self.final(x04)
    
########################
# Complete Unet++
########################

# ---- UNet++ ----
class UNetPPcomplete(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        filters = [base, base*2, base*4, base*8, base*16]

        self.conv00 = ConvBlock3(in_ch, filters[0])
        self.conv10 = ConvBlock3(filters[0], filters[1])
        self.conv20 = ConvBlock3(filters[1], filters[2])
        self.conv30 = ConvBlock3(filters[2], filters[3])
        self.conv40 = ConvBlock3(filters[3], filters[4])

        self.up01 = Up(filters[1], filters[0])
        self.up11 = Up(filters[2], filters[1])
        self.up21 = Up(filters[3], filters[2])
        self.up31 = Up(filters[4], filters[3])

        self.conv01 = ConvBlock3(filters[0] + filters[0], filters[0])
        self.conv11 = ConvBlock3(filters[1] + filters[1], filters[1])
        self.conv21 = ConvBlock3(filters[2] + filters[2], filters[2])
        self.conv31 = ConvBlock3(filters[3] + filters[3], filters[3])

        self.up02 = Up(filters[1], filters[0])
        self.up12 = Up(filters[2], filters[1])
        self.up22 = Up(filters[3], filters[2])
        self.cut = lambda a,b: a[:, :, :b.size(2), :b.size(3)]

        self.conv02 = ConvBlock3(filters[0]*3, filters[0])
        self.conv12 = ConvBlock3(filters[1]*3, filters[1])
        self.conv22 = ConvBlock3(filters[2]*3, filters[2])

        self.up03 = Up(filters[1], filters[0])
        self.conv03 = ConvBlock3(filters[0]*4, filters[0])

        self.up04 = Up(filters[1], filters[0])
        self.conv04 = ConvBlock3(filters[0]*5, filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, 1)

        self.up13 = Up(filters[2], filters[1])
        self.up23 = Up(filters[3], filters[2])

        self.conv13 = ConvBlock3(filters[1]*4, filters[1])
        self.conv23 = ConvBlock3(filters[2]*4, filters[2])

        self.up14 = Up(filters[1], filters[0])
        self.conv14 = ConvBlock3(filters[0]*6, filters[0])

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(F.max_pool2d(x00, 2))
        x20 = self.conv20(F.max_pool2d(x10, 2))
        x30 = self.conv30(F.max_pool2d(x20, 2))
        x40 = self.conv40(F.max_pool2d(x30, 2))

        x01 = self.conv01(torch.cat([x00, self.up01(x10)], 1))
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], 1))

        x21 = self.conv21(torch.cat([x20, self.up21(x30)], 1))
        x31 = self.conv31(torch.cat([x30, self.up31(x40)], 1))

        x02 = self.conv02(torch.cat([x00, self.cut(self.up02(x11), x00), x01], 1))
        x12 = self.conv12(torch.cat([x10, self.cut(self.up12(x21), x10), x11], 1))
        x22 = self.conv22(torch.cat([x20, self.cut(self.up22(x31), x20), x21], 1))

        x03 = self.conv03(torch.cat([x00, self.up03(x12), x01, x02], 1))
        x04 = self.conv04(torch.cat([x00, self.cut(self.up04(x12), x00), x01, x02, x03], 1))

        x13 = self.conv13(torch.cat([x10,self.cut(self.up13(x22), x10),x11,x12], 1))
        x23 = self.conv23(torch.cat([x20,self.cut(self.up23(x31), x20),x21,x22], 1))
        x14 = self.conv14(torch.cat([x00,self.cut(self.up14(x13), x00),x01,x02,x03,x04], 1))

        #return torch.sigmoid(self.final(x14))
        return self.final(x14)

###############################################
# DeepLabV3+
###############################################

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, stride=stride)
        self.point = nn.Conv2d(in_ch, out_ch, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.branch1 = nn.Conv2d(ch, ch, 1)
        self.branch2 = nn.Conv2d(ch, ch, 3, padding=6, dilation=6)
        self.branch3 = nn.Conv2d(ch, ch, 3, padding=12, dilation=12)
        self.branch4 = nn.Conv2d(ch, ch, 3, padding=18, dilation=18)

        self.project = nn.Sequential(
            nn.Conv2d(ch*4, ch, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.project(out)

class DeepLabV3PlusLight(nn.Module):
    def __init__(self, input_channels=3, base_ch=32):
        super().__init__()
        c = base_ch

        # Lightweight encoder
        self.conv1 = DepthwiseSeparableConv(input_channels, c, stride=1)
        self.conv2 = DepthwiseSeparableConv(c, c*2, stride=2)
        self.conv3 = DepthwiseSeparableConv(c*2, c*4, stride=2)

        # ASPP
        self.aspp = ASPP(c*4)

        # Decoder
        self.up = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.conv_dec = nn.Sequential(
            nn.Conv2d(c*2 + c*2, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.final = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        a = self.aspp(c3)
        u = self.up(a)

        u = torch.cat([u, c2], dim=1)
        d = self.conv_dec(u)

        out = self.up2(d)
        out = self.final(out)
        return out

###############################################
# faithfull DeepLabV3+
###############################################
    
class ConvBNReLUnew(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ASPPnew(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()

        self.branch1 = ConvBNReLUnew(in_ch, out_ch, kernel_size=1, padding=0)

        self.branch2 = ConvBNReLUnew(in_ch, out_ch, dilation=6, padding=6)
        self.branch3 = ConvBNReLUnew(in_ch, out_ch, dilation=12, padding=12)
        self.branch4 = ConvBNReLUnew(in_ch, out_ch, dilation=18, padding=18)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLUnew(in_ch, out_ch, kernel_size=1, padding=0)
        )

        self.project = ConvBNReLUnew(out_ch * 5, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        h, w = x.shape[2:]

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)

class Encodernew(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        # Low-level features (used by decoder)
        self.layer1 = nn.Sequential(
            ConvBNReLUnew(in_ch, 64, stride=2),
            ConvBNReLUnew(64, 64),
            ConvBNReLUnew(64, 64)
        )

        # Downsample
        self.layer2 = nn.Sequential(
            ConvBNReLUnew(64, 128, stride=2),
            ConvBNReLUnew(128, 128)
        )

        # Downsample to OS=16
        self.layer3 = nn.Sequential(
            ConvBNReLUnew(128, 256, stride=2),
            ConvBNReLUnew(256, 256)
        )

        # Dilated convs (no more downsampling)
        self.layer4 = nn.Sequential(
            ConvBNReLUnew(256, 512, dilation=2, padding=2),
            ConvBNReLUnew(512, 512, dilation=2, padding=2)
        )

    def forward(self, x):
        low = self.layer1(x)    # low-level features
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low

class Decodernew(nn.Module):
    def __init__(self, low_ch=64, aspp_ch=256, out_ch=256):
        super().__init__()

        self.low_proj = ConvBNReLUnew(low_ch, 48, kernel_size=1, padding=0)

        self.conv = nn.Sequential(
            ConvBNReLUnew(aspp_ch + 48, out_ch),
            ConvBNReLUnew(out_ch, out_ch)
        )

    def forward(self, aspp_feat, low_feat):
        low_feat = self.low_proj(low_feat)

        aspp_feat = F.interpolate(
            aspp_feat,
            size=low_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        x = torch.cat([aspp_feat, low_feat], dim=1)
        return self.conv(x)

class DeepLabV3Plusnew(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        self.encoder = Encodernew(in_channels)
        self.aspp = ASPPnew(512)
        self.decoder = Decodernew()

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        enc, low = self.encoder(x)
        x = self.aspp(enc)
        x = self.decoder(x, low)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return self.classifier(x)

###############################################
# DenseU-net
###############################################

class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth=16, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(ch, growth, 3, padding=1),
                nn.ReLU(inplace=True)
            ))
            ch += growth
        self.out_ch = ch

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            new = layer(torch.cat(feats, dim=1))
            feats.append(new)
        return torch.cat(feats, dim=1)


class Up2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    def forward(self, x): return self.up(x)


class DenseUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.db1 = DenseBlock(in_ch, growth=base//4)
        self.db2 = DenseBlock(self.db1.out_ch, growth=base//2)
        self.db3 = DenseBlock(self.db2.out_ch, growth=base)

        self.pool = nn.MaxPool2d(2)

        self.up1 = Up2(self.db3.out_ch, self.db2.out_ch)
        self.up2 = Up2(self.db2.out_ch, self.db1.out_ch)

        self.conv_up1 = nn.Conv2d(self.db2.out_ch * 2, self.db2.out_ch, 3, padding=1)
        self.conv_up2 = nn.Conv2d(self.db1.out_ch * 2, self.db1.out_ch, 3, padding=1)

        self.final = nn.Conv2d(self.db1.out_ch, out_ch, 1)

    def forward(self, x):
        x1 = self.db1(x)
        x2 = self.db2(self.pool(x1))
        x3 = self.db3(self.pool(x2))

        u1 = self.up1(x3)
        u1 = torch.cat([x2, u1], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([x1, u2], dim=1)
        u2 = self.conv_up2(u2)

        #return torch.sigmoid(self.final(u2))
        return self.final(u2)

###############################################
# HRNetSmallSeg
###############################################

# ---------------------------
# Basic residual block
# ---------------------------
class BasicResidual(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc = self.shortcut(x)
        return self.relu(out + sc)

# ---------------------------
# Small HR module
# ---------------------------
class HRModuleSmall(nn.Module):
    """
    HRModule that fuses multiple resolution branches.
    branches: list of channel counts per branch
    """
    def __init__(self, branches):
        super().__init__()
        self.num_branches = len(branches)
        self.branches = nn.ModuleList([
            nn.Sequential(BasicResidual(ch, ch), BasicResidual(ch, ch)) for ch in branches
        ])

        # Fusion layers (all 1x1 convs to adjust channels)
        self.fuse_layers = nn.ModuleList()
        for i in range(self.num_branches):
            fuse_i = nn.ModuleList()
            for j in range(self.num_branches):
                if i == j:
                    fuse_i.append(nn.Identity())
                else:
                    fuse_i.append(nn.Conv2d(branches[j], branches[i], kernel_size=1))
            self.fuse_layers.append(fuse_i)

    def forward(self, x_branches):
        # Process each branch
        x_processed = [self.branches[i](x_branches[i]) for i in range(self.num_branches)]

        # Fuse
        x_fused = []
        for i in range(self.num_branches):
            target_size = x_processed[i].shape[2:]  # spatial size to match
            y = 0
            for j in range(self.num_branches):
                to_add = self.fuse_layers[i][j](x_processed[j]) if i != j else x_processed[j]
                if to_add.shape[2:] != target_size:
                    to_add = F.interpolate(to_add, size=target_size, mode='bilinear', align_corners=False)
                y = to_add if isinstance(y, int) else y + to_add
            x_fused.append(F.relu(y))
        return x_fused

# ---------------------------
# Small HRNet for segmentation
# ---------------------------
class HRNetSmallSeg(nn.Module):
    def __init__(self, input_channels=3, base_ch=32, num_modules=2):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(input_channels, base_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.relu = nn.ReLU(inplace=True)

        # Initial branches
        self.layer1 = BasicResidual(base_ch, base_ch)  # high-res branch
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),
            BasicResidual(base_ch*2, base_ch*2)
        )  # mid-res
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True),
            BasicResidual(base_ch*4, base_ch*4)
        )  # low-res

        branches = [base_ch, base_ch*2, base_ch*4]
        self.hr_modules = nn.ModuleList([HRModuleSmall(branches) for _ in range(num_modules)])

        # Fusion to high resolution
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(sum(branches), base_ch*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(base_ch, 1, kernel_size=1)  # output segmentation

    def forward(self, x):
        # Stem + initial branches
        x0 = self.relu(self.bn1(self.conv1(x)))  # high-res
        b0 = self.layer1(x0)
        b1 = self.layer2(x0)  # mid-res
        b2 = self.layer3(b1)  # low-res

        branches = [b0, b1, b2]
        for mod in self.hr_modules:
            branches = mod(branches)

        # Fuse all to high-res
        b1_up = F.interpolate(branches[1], size=branches[0].shape[2:], mode='bilinear', align_corners=False)
        b2_up = F.interpolate(branches[2], size=branches[0].shape[2:], mode='bilinear', align_corners=False)

        cat = torch.cat([branches[0], b1_up, b2_up], dim=1)
        fused = self.conv_fuse(cat)
        out = self.final(fused)
        return out

###############################################
# SegFormer
###############################################

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3)
        self.norm = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.norm(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0,2,3,1).reshape(B, -1, C)
        res = x_flat
        x_flat = self.norm1(x_flat)
        attn_out,_ = self.attn(x_flat, x_flat, x_flat)
        x_flat = res + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.reshape(B, H, W, C).permute(0,3,1,2)


class SegFormerMini(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, embed=32):
        super().__init__()
        self.patch = OverlapPatchEmbed(in_ch, embed, stride=4)
        self.block = TransformerBlock(embed, heads=4)
        self.up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.seghead = nn.Conv2d(embed, out_ch, 1)

    def forward(self, x):
        x = self.patch(x)
        x = self.block(x)
        x = self.up(x)
        #return torch.sigmoid(self.seghead(x))
        return self.seghead(x)

###############################################
# faithfull SegFormer
###############################################

class OverlapPatchEmbednew(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3, stride=stride, padding=1
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W
    
class EfficientAttentionnew(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        if self.sr_ratio > 1:
            x_ = x.transpose(1,2).reshape(-1, x.shape[2], H, W)
            x_ = self.sr(x_).flatten(2).transpose(1,2)
            x_ = self.norm(x_)
        else:
            x_ = x

        out,_ = self.attn(x, x_, x_)
        return out

class MixFFNnew(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.dwconv = nn.Conv2d(
            dim * mlp_ratio,
            dim * mlp_ratio,
            3, padding=1,
            groups=dim * mlp_ratio
        )
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = x.transpose(1,2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1,2)
        return self.fc2(x)

class MiTBlocknew(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttentionnew(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFNnew(dim)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class MiTEncodernew(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        self.embed_dims = [64, 128, 320, 512]
        self.depths = [2, 2, 2, 2]
        self.heads = [1, 2, 5, 8]
        self.sr_ratios = [8, 4, 2, 1]

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()

        prev_ch = in_ch
        for i in range(4):
            self.patch_embeds.append(
                OverlapPatchEmbednew(prev_ch, self.embed_dims[i], stride=2 if i>0 else 4)
            )
            self.blocks.append(
                nn.ModuleList([
                    MiTBlocknew(self.embed_dims[i], self.heads[i], self.sr_ratios[i])
                    for _ in range(self.depths[i])
                ])
            )
            prev_ch = self.embed_dims[i]

    def forward(self, x):
        features = []
        for pe, blocks in zip(self.patch_embeds, self.blocks):
            x, H, W = pe(x)
            for blk in blocks:
                x = blk(x, H, W)
            feat = x.transpose(1,2).reshape(-1, x.shape[2], H, W)
            features.append(feat)
            x = feat
        return features

class SegFormerDecodernew(nn.Module):
    def __init__(self, embed_dims, num_classes):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(d, 256, 1) for d in embed_dims
        ])
        self.fuse = nn.Conv2d(256*4, 256, 1)
        self.cls = nn.Conv2d(256, num_classes, 1)

    def forward(self, feats):
        size = feats[0].shape[2:]
        outs = []
        for f, p in zip(feats, self.proj):
            f = p(f)
            f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
            outs.append(f)
        x = self.fuse(torch.cat(outs, dim=1))
        return self.cls(x)

class SegFormernew(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.encoder = MiTEncodernew(in_ch)
        self.decoder = SegFormerDecodernew(
            embed_dims=[64,128,320,512],
            num_classes=num_classes
        )

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        out = F.interpolate(
            out,
            size=x.shape[2:],      
            mode="bilinear",
            align_corners=False
        )
        return out

###############################################
# BiSeNet
###############################################

# ----------------------------------------------
# Spatial Path (preserve resolution)
# ----------------------------------------------
class SpatialPath(nn.Module):
    def __init__(self, in_ch=3, ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

        self.conv3 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

# ----------------------------------------------
# Simple Context Path (compact)
# ----------------------------------------------
class ContextPath(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        # small CNN for context
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


# ----------------------------------------------
# Attention Fusion Module
# ----------------------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, sp_ch, ct_ch, out_ch=128):
        super().__init__()
        self.conv = nn.Conv2d(sp_ch + ct_ch, out_ch, 1)

        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, sp, ct):
        x = torch.cat([sp, ct], dim=1)
        x = self.conv(x)
        a = self.att(x)
        return x * a + x

# ----------------------------------------------
# Full BiSeNet-Light
# ----------------------------------------------
class BiSeNetLight(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.spatial = SpatialPath(input_channels)
        self.context = ContextPath(input_channels)
        self.fuse = AttentionFusion(64, 256, 128)

        self.final = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        sp = self.spatial(x)
        ct = self.context(x)

        ct_up = F.interpolate(ct, size=sp.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fuse(sp, ct_up)

        out = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
        return self.final(out)
    
###############################################
# BiSeNet corrected
###############################################

class ContextPathnew(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # ×16
        self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # ×32

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        gp = self.global_pool(x)
        x = x * self.att(gp)
        return x
    
class BiSeNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.spatial = SpatialPath(input_channels)
        self.context = ContextPathnew(input_channels)
        self.fuse = AttentionFusion(64, 256, 128)

        self.final = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        sp = self.spatial(x)
        ct = self.context(x)

        ct_up = F.interpolate(ct, size=sp.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fuse(sp, ct_up)

        out = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
        return self.final(out)

###############################################
# U-net3+
###############################################

# -------------------------
# Basic conv block
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# -------------------------
# UNet3+ Model
# -------------------------
class UNet3Plus(nn.Module):
    def __init__(self, input_channels=3, base_ch=32):
        super().__init__()
        c = base_ch

        # Encoder
        self.conv1 = ConvBNReLU(input_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBNReLU(c, c*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBNReLU(c*2, c*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBNReLU(c*4, c*8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ConvBNReLU(c*8, c*16)

        # Decoder (UNet 3+ style: aggregate all encoder outputs for each stage)
        self.up4 = nn.ConvTranspose2d(c*16, c*8, 2, stride=2)
        self.conv4d = ConvBNReLU(c*8 + c*8 + c*4 + c*2 + c, c*8)

        self.up3 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.conv3d = ConvBNReLU(c*4 + c*4 + c*2 + c, c*4)

        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.conv2d = ConvBNReLU(c*2 + c*2 + c, c*2)

        self.up1 = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.conv1d = ConvBNReLU(c + c, c)

        # Final
        self.final = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)      # (B, c, H, W)
        x2 = self.conv2(self.pool1(x1))  # (B, 2c, H/2, W/2)
        x3 = self.conv3(self.pool2(x2))  # (B, 4c, H/4, W/4)
        x4 = self.conv4(self.pool3(x3))  # (B, 8c, H/8, W/8)
        x5 = self.conv5(self.pool4(x4))  # (B,16c, H/16, W/16)

        # Decoder stage 4
        u4 = self.up4(x5)
        u4 = self._resize_to(u4, x4)
        cat4 = torch.cat([u4, x4, self._resize_to(x3, x4), self._resize_to(x2, x4), self._resize_to(x1, x4)], dim=1)
        d4 = self.conv4d(cat4)

        # Decoder stage 3
        u3 = self.up3(d4)
        u3 = self._resize_to(u3, x3)
        cat3 = torch.cat([u3, x3, self._resize_to(x2, x3), self._resize_to(x1, x3)], dim=1)
        d3 = self.conv3d(cat3)

        # Decoder stage 2
        u2 = self.up2(d3)
        u2 = self._resize_to(u2, x2)
        cat2 = torch.cat([u2, x2, self._resize_to(x1, x2)], dim=1)
        d2 = self.conv2d(cat2)

        # Decoder stage 1
        u1 = self.up1(d2)
        u1 = self._resize_to(u1, x1)
        cat1 = torch.cat([u1, x1], dim=1)
        d1 = self.conv1d(cat1)

        out = self.final(d1)
        return out
    
    def _resize_to(self, src, target):
        _, _, H, W = target.shape
        return F.interpolate(src, size=(H, W), mode='bilinear', align_corners=False)

############################################
# Transfo2: Transformer with bilinear decoder and small patches 
############################################

def resize_pos_emb(pos_emb, target_grid_hw):
    # pos_emb: (1, N, C) where N = H'*W'
    # target_grid_hw: (Ht, Wt)
    B1, N, C = pos_emb.shape
    Ht, Wt = target_grid_hw
    Hs = Ws = int(math.sqrt(N))
    assert Hs * Ws == N, "pos_emb isn't square grid"
    pos = pos_emb[0].permute(1,0).view(C, Hs, Ws).unsqueeze(0)  # (1,C,Hs,Ws)
    pos = F.interpolate(pos, size=(Ht, Wt), mode='bicubic', align_corners=False)
    pos = pos.view(1, C, Ht*Wt).permute(0,2,1)  # (1, Ht*Wt, C)
    return pos

# Simple patch embed (conv)
class PatchEmbedConv(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        # conv with kernel=patch_size and stride=patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0,2,1)  # (B, num_patches, embed_dim)
        return x, (H, W)

# Transformer Encoder block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

# SimpleViTSeg (small patch + bilinear decoder)
class SimpleViTSeg(nn.Module):
    def __init__(self, input_shape=(256,256,3), patch_size=4, embedding_dim=256, num_heads=4, transformer_layers=6, mlp_dim=512, num_classes=1):
        super().__init__()
        img_size = input_shape[0]
        in_ch = input_shape[2]
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedConv(in_ch=in_ch, embed_dim=embedding_dim, patch_size=patch_size)
        # initial pos emb (square grid)
        grid_size = (img_size // patch_size, img_size // patch_size)
        num_patches = grid_size[0] * grid_size[1]
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, embedding_dim) * 0.02)

        self.transformer = nn.ModuleList([TransformerEncoderBlock(embedding_dim, num_heads, mlp_dim) for _ in range(transformer_layers)])

        # Decoder: reshape -> bilinear upsample to original size, then conv refinement
        self.refine = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embedding_dim//2, embedding_dim//4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embedding_dim//4, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # if input not divisible by patch_size, resize input to divisible via interpolation
        target_h = (H // self.patch_size) * self.patch_size
        target_w = (W // self.patch_size) * self.patch_size
        if target_h != H or target_w != W:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            H, W = target_h, target_w

        x_tokens, (gh, gw) = self.patch_embed(x)  # (B, N, D)
        # Resize pos emb if needed
        if x_tokens.shape[1] != self.pos_emb.shape[1]:
            pos = resize_pos_emb(self.pos_emb, (gh, gw)).to(x_tokens.device)
        else:
            pos = self.pos_emb
        x_tokens = x_tokens + pos

        for blk in self.transformer:
            x_tokens = blk(x_tokens)

        # reshape to spatial
        x_sp = x_tokens.permute(0,2,1).contiguous().view(B, -1, gh, gw)  # (B, D, gh, gw)
        # upsample to original image resolution
        scale = self.patch_size
        x_up = F.interpolate(x_sp, size=(H, W), mode='bilinear', align_corners=False)
        out = self.refine(x_up)  # (B, num_classes, H, W)
        return out 

#################################
# CBAM Unet3+
#################################

class CBAMBlocknew(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_att = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att

# UNet3+ with CBAM on skip connections
class UNet3Plus_CBAM(nn.Module):
    def __init__(self, input_channels=3, base_ch=32):
        super().__init__()
        c = base_ch

        # Encoder
        self.conv1 = ConvBNReLU(input_channels, c)
        self.conv2 = ConvBNReLU(c, c * 2)
        self.conv3 = ConvBNReLU(c * 2, c * 4)
        self.conv4 = ConvBNReLU(c * 4, c * 8)
        self.conv5 = ConvBNReLU(c * 8, c * 16)

        self.pool = nn.MaxPool2d(2)

        # CBAM blocks (only for skips)
        self.cbam1 = CBAMBlocknew(c)
        self.cbam2 = CBAMBlocknew(c * 2)
        self.cbam3 = CBAMBlocknew(c * 4)
        self.cbam4 = CBAMBlocknew(c * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2)
        self.conv4d = ConvBNReLU(c * (8 + 8 + 4 + 2 + 1), c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.conv3d = ConvBNReLU(c * (4 + 4 + 2 + 1), c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.conv2d = ConvBNReLU(c * (2 + 2 + 1), c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.conv1d = ConvBNReLU(c * 2, c)

        self.final = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        # Decoder stage 4
        u4 = self._resize_to(self.up4(x5), x4)
        cat4 = torch.cat([
            u4,
            self.cbam4(x4),
            self._resize_to(self.cbam3(x3), x4),
            self._resize_to(self.cbam2(x2), x4),
            self._resize_to(self.cbam1(x1), x4)
        ], dim=1)
        d4 = self.conv4d(cat4)

        # Decoder stage 3
        u3 = self._resize_to(self.up3(d4), x3)
        cat3 = torch.cat([
            u3,
            self.cbam3(x3),
            self._resize_to(self.cbam2(x2), x3),
            self._resize_to(self.cbam1(x1), x3)
        ], dim=1)
        d3 = self.conv3d(cat3)

        # Decoder stage 2
        u2 = self._resize_to(self.up2(d3), x2)
        cat2 = torch.cat([
            u2,
            self.cbam2(x2),
            self._resize_to(self.cbam1(x1), x2)
        ], dim=1)
        d2 = self.conv2d(cat2)

        # Decoder stage 1
        u1 = self._resize_to(self.up1(d2), x1)
        d1 = self.conv1d(torch.cat([u1, self.cbam1(x1)], dim=1))

        return self.final(d1)

    def _resize_to(self, src, tgt):
        return F.interpolate(src, size=tgt.shape[2:], mode="bilinear", align_corners=False)
    
#################################
# Attention Unet3+
#################################

# Attention Gate 
class AttentionGatenew(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.theta_x = nn.Conv2d(in_ch, inter_ch, kernel_size=2, stride=2, bias=False)
        self.phi_g = nn.Conv2d(gating_ch, inter_ch, kernel_size=1, bias=True)
        self.f = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, g):
        theta_x = self.theta_x(x)
        phi_g = F.interpolate(self.phi_g(g), size=theta_x.shape[2:], mode='bilinear', align_corners=False)
        psi = self.f(theta_x + phi_g)
        psi = self.up_sample(psi)
        return x * psi

# UNet3+ with Attention Gates
class UNet3Plus_AG(nn.Module):
    def __init__(self, input_channels=3, base_ch=32):
        super().__init__()
        c = base_ch

        # Encoder
        self.conv1 = ConvBNReLU(input_channels, c)
        self.conv2 = ConvBNReLU(c, c * 2)
        self.conv3 = ConvBNReLU(c * 2, c * 4)
        self.conv4 = ConvBNReLU(c * 4, c * 8)
        self.conv5 = ConvBNReLU(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2)
        self.ag4 = AttentionGatenew(
            in_ch=c * (8 + 4 + 2 + 1),
            gating_ch=c * 8,
            inter_ch=c * 4
        )
        self.conv4d = ConvBNReLU(c * (8 + 8 + 4 + 2 + 1), c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.ag3 = AttentionGatenew(
            in_ch=c * (4 + 2 + 1),
            gating_ch=c * 4,
            inter_ch=c * 2
        )
        self.conv3d = ConvBNReLU(c * (4 + 4 + 2 + 1), c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.ag2 = AttentionGatenew(
            in_ch=c * (2 + 1),
            gating_ch=c * 2,
            inter_ch=c
        )
        self.conv2d = ConvBNReLU(c * (2 + 2 + 1), c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.conv1d = ConvBNReLU(c * 2, c)

        self.final = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        # Decoder stage 4
        u4 = self._resize_to(self.up4(x5), x4)
        skips4 = torch.cat([
            x4,
            self._resize_to(x3, x4),
            self._resize_to(x2, x4),
            self._resize_to(x1, x4)
        ], dim=1)
        skips4 = self.ag4(skips4, u4)
        d4 = self.conv4d(torch.cat([u4, skips4], dim=1))

        # Decoder stage 3
        u3 = self._resize_to(self.up3(d4), x3)
        skips3 = torch.cat([
            x3,
            self._resize_to(x2, x3),
            self._resize_to(x1, x3)
        ], dim=1)
        skips3 = self.ag3(skips3, u3)
        d3 = self.conv3d(torch.cat([u3, skips3], dim=1))

        # Decoder stage 2
        u2 = self._resize_to(self.up2(d3), x2)
        skips2 = torch.cat([x2, self._resize_to(x1, x2)], dim=1)
        skips2 = self.ag2(skips2, u2)
        d2 = self.conv2d(torch.cat([u2, skips2], dim=1))

        # Decoder stage 1
        u1 = self._resize_to(self.up1(d2), x1)
        d1 = self.conv1d(torch.cat([u1, x1], dim=1))

        return self.final(d1)

    def _resize_to(self, src, tgt):
        return F.interpolate(src, size=tgt.shape[2:], mode="bilinear", align_corners=False)

#################################
# Deep Dense-HRNet 
#################################

# Dense Block
class DenseBlocknew(nn.Module):
    def __init__(self, in_ch, growth_rate=16, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, growth_rate, kernel_size=3, padding=1, bias=False)
                )
            )
            ch += growth_rate

        # compress back to in_ch
        self.transition = nn.Conv2d(ch, in_ch, kernel_size=1, bias=False)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return self.transition(torch.cat(features, dim=1))

# HR Module with Dense Blocks
class HRModuleDense(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.num_branches = len(branches)

        self.branches = nn.ModuleList([
            nn.Sequential(
                DenseBlocknew(ch),
                DenseBlocknew(ch)
            ) for ch in branches
        ])

        self.fuse_layers = nn.ModuleList()
        for i in range(self.num_branches):
            fuse_i = nn.ModuleList()
            for j in range(self.num_branches):
                if i == j:
                    fuse_i.append(nn.Identity())
                else:
                    fuse_i.append(nn.Conv2d(branches[j], branches[i], 1))
            self.fuse_layers.append(fuse_i)

    def forward(self, x_branches):
        x_proc = [self.branches[i](x_branches[i]) for i in range(self.num_branches)]

        out = []
        for i in range(self.num_branches):
            target_size = x_proc[i].shape[2:]
            y = 0
            for j in range(self.num_branches):
                t = self.fuse_layers[i][j](x_proc[j]) if i != j else x_proc[j]
                if t.shape[2:] != target_size:
                    t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
                y = t if isinstance(y, int) else y + t
            out.append(F.relu(y))
        return out

# HRNet Small Dense – Deeper 
class HRNetSmallDenseSeg(nn.Module):
    def __init__(self, input_channels=3, base_ch=32, num_modules=3):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, base_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.relu = nn.ReLU(inplace=True)

        # Initial branches
        self.b0 = DenseBlocknew(base_ch)
        self.b1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            DenseBlocknew(base_ch * 2)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            DenseBlocknew(base_ch * 4)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
            DenseBlocknew(base_ch * 8)
        )

        branches = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.hr_modules = nn.ModuleList([
            HRModuleDense(branches) for _ in range(num_modules)
        ])

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(sum(branches), base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        b0 = self.b0(x)
        b1 = self.b1(x)
        b2 = self.b2(b1)
        b3 = self.b3(b2)

        branches = [b0, b1, b2, b3]
        for m in self.hr_modules:
            branches = m(branches)

        ups = [
            branches[0],
            F.interpolate(branches[1], size=branches[0].shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(branches[2], size=branches[0].shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(branches[3], size=branches[0].shape[2:], mode='bilinear', align_corners=False),
        ]

        fused = self.conv_fuse(torch.cat(ups, dim=1))
        return self.final(fused)

##############################
# Unet BiSeNet auxiliary path
##############################

# BiSeNet-style auxiliary path on input image
class BiSeNetInputPath(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()

        self.spatial = SpatialPath(in_ch=in_ch, ch=out_ch)
        self.context = ContextPathnew(in_ch=in_ch)

        self.context_reduce = nn.Conv2d(256, out_ch, 1)

        self.fusion = AttentionFusion(
            sp_ch=out_ch,
            ct_ch=out_ch,
            out_ch=out_ch
        )

    def forward(self, x):
        sp = self.spatial(x)            # low-res spatial
        ct = self.context(x)            # very low-res context

        ct = self.context_reduce(ct)
        ct = F.interpolate(ct, size=sp.shape[2:], mode='bilinear', align_corners=False)

        fused = self.fusion(sp, ct)

        # Bring back to full resolution for final concat
        fused = F.interpolate(fused, size=x.shape[2:], mode='bilinear', align_corners=False)
        return fused
    
class SimpleUNet_BiSeLateFusion(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.drop4 = nn.Dropout(0.3)

        self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        # BiSeNet auxiliary path on input image
        self.bisenet_input = BiSeNetInputPath(
            in_ch=input_channels,
            out_ch=64
        )

        # Input channels: u5 (64) + c1 (64) + bisenet (64) = 192
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.drop5 = nn.Dropout(0.3)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.drop1(self.pool1(c1))

        c2 = self.conv2(p1)
        p2 = self.drop2(self.pool2(c2))

        c3 = self.conv3(p2)

        # Decoder
        u4 = self.up4(c3)
        u4 = torch.cat([u4, c2], dim=1)
        c4 = self.drop4(self.conv4(u4))

        u5 = self.up5(c4)

        # BiSeNet-style late fusion
        bi_feat = self.bisenet_input(x)

        u5 = torch.cat([u5, c1, bi_feat], dim=1)
        c5 = self.drop5(self.conv5(u5))

        return self.final(c5)
    
################################
# JPU enhanced Attention Unet
################################

class JPUModulenew(nn.Module):
    def __init__(self, in_channels, width=64):
        """
        in_channels: list of channel counts (len = 2 or 3)
        """
        super().__init__()
        self.num_inputs = len(in_channels)

        self.convs = nn.ModuleList([
            nn.Conv2d(ch, width, 3, padding=1)
            for ch in in_channels
        ])

        # Parallel dilated convolutions (unchanged)
        self.dilation_convs = nn.ModuleList([
            nn.Conv2d(self.num_inputs * width, width, 3, padding=d, dilation=d)
            for d in [1, 2, 4, 8]
        ])

    def forward(self, feats):
        assert len(feats) == self.num_inputs

        feat0 = self.convs[0](feats[0])
        base_size = feat0.shape[2:]

        proj_feats = [feat0]
        for i in range(1, self.num_inputs):
            f = self.convs[i](feats[i])
            f = F.interpolate(f, size=base_size, mode='bilinear', align_corners=False)
            proj_feats.append(f)

        feat = torch.cat(proj_feats, dim=1)

        outs = [conv(feat) for conv in self.dilation_convs]
        return torch.cat(outs, dim=1)  # (B, 4*width, H, W)
    
def jpu_reduce(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1)

class AttentionUNet_JPU(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super().__init__()
        f = base_filters

        # Encoder (unchanged)
        self.conv1 = ConvBlock2(input_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = ConvBlock2(f, f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = ConvBlock2(f*2, f*4)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = ConvBlock2(f*4, f*8)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(0.3)

        self.conv5 = ConvBlock2(f*8, f*16)

        # JPU modules for skips
        self.jpu_c4 = JPUModulenew([f*4, f*8], width=f)
        self.jpu_c3 = JPUModulenew([f*2, f*4, f*8], width=f)
        self.jpu_c2 = JPUModulenew([f, f*2, f*4], width=f)
        self.jpu_c1 = JPUModulenew([f, f*2], width=f)

        # Reduce JPU outputs
        self.red4 = jpu_reduce(4*f, f*8)
        self.red3 = jpu_reduce(4*f, f*4)
        self.red2 = jpu_reduce(4*f, f*2)
        self.red1 = jpu_reduce(4*f, f)

        # Decoder + attention (unchanged)
        self.up6 = nn.ConvTranspose2d(f*16, f*8, 2, 2)
        self.att6 = AttentionGate(f*8, f*8, f*4)
        self.conv6 = ConvBlock2(f*16, f*8)

        self.up7 = nn.ConvTranspose2d(f*8, f*4, 2, 2)
        self.att7 = AttentionGate(f*4, f*4, f*2)
        self.conv7 = ConvBlock2(f*8, f*4)

        self.up8 = nn.ConvTranspose2d(f*4, f*2, 2, 2)
        self.att8 = AttentionGate(f*2, f*2, f)
        self.conv8 = ConvBlock2(f*4, f*2)

        self.up9 = nn.ConvTranspose2d(f*2, f, 2, 2)
        self.att9 = AttentionGate(f, f, max(f//2, 8))
        self.conv9 = ConvBlock2(f*2, f)

        self.final = nn.Conv2d(f, 1, 1)

    def _resize_to(self, src, tgt):
        return F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))

        u6 = self._resize_to(self.up6(c5), c4)
        j4 = self.red4(self.jpu_c4([c3, c4]))
        j4 = self._resize_to(j4, u6) 
        c4_att = self.att6(j4, u6)
        c6 = self.conv6(torch.cat([u6, c4_att], dim=1))

        u7 = self._resize_to(self.up7(c6), c3)
        j3 = self.red3(self.jpu_c3([c2, c3, c4]))
        j3 = self._resize_to(j3, u7)   
        c3_att = self.att7(j3, u7)
        c7 = self.conv7(torch.cat([u7, c3_att], dim=1))

        u8 = self._resize_to(self.up8(c7), c2)
        j2 = self.red2(self.jpu_c2([c1, c2, c3]))
        j2 = self._resize_to(j2, u8)   
        c2_att = self.att8(j2, u8)
        c8 = self.conv8(torch.cat([u8, c2_att], dim=1))

        u9 = self._resize_to(self.up9(c8), c1)
        j1 = self.red1(self.jpu_c1([c1, c2]))
        j1 = self._resize_to(j1, u9)       
        c1_att = self.att9(j1, u9)
        c9 = self.conv9(torch.cat([u9, c1_att], dim=1))

        return self.final(c9) 
