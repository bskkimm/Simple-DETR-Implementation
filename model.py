import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from config import dataset_config
from config import model_config



class Backbone(nn.Module):
    def __init__(self,
                 backbone_dim=model_config['backbone_dim'],
                 backbone_model=model_config['backbone_model'],
                 model_dim=model_config['model_dim']):
        super().__init__()
        # Load the pre-trained resnet50 model without final FC layer and tracking gradients
        if backbone_model == 50:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, norm_layer=torchvision.ops.FrozenBatchNorm2d)
            # Remove the final fully connected layer
        elif backbone_model == 101:
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT, norm_layer=torchvision.ops.FrozenBatchNorm2d)
            # Remove the final fully connected layer

        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        if model_config['freeze_backbone']:
            for param in self.resnet.parameters():
                param.requires_grad = False


        self.embed_proj = nn.Conv2d(in_channels=backbone_dim,
                                   out_channels=model_dim,
                                   kernel_size=1)


    def forward(self, x): 
        # x is the initial batch input (B, 3, H ,W)
        x = self.resnet(x) # (B, 2048, H', W')
        x = self.embed_proj(x) # (B, embed_dim, H', W')
        x = x.permute(0, 2, 3, 1) # (B, H', W', embed_dim)
        x = x.flatten(1, 2)  # (B, H'*W', embed_dim)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, model_dim=model_config['model_dim']):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, model_dim))  # (max_len, D)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.size()
        return self.pe[:N].unsqueeze(0).repeat(B, 1, 1)  # (B, N, D)


class EncoderLayer(nn.Module):
    def __init__(self,
                 model_dim=model_config['model_dim'],
                 encoder_heads=model_config['encoder_heads'],
                 mlp_inner_dim=model_config['mlp_inner_dim']):
        super().__init__()
        # Attention layer
        self.self_attn = nn.MultiheadAttention(model_dim, encoder_heads)
        self.norm_attn = nn.LayerNorm(model_dim)

        # MLP layer
        self.linear1 = nn.Linear(model_dim, mlp_inner_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(mlp_inner_dim, model_dim)
        self.norm_mlp = nn.LayerNorm(model_dim)

    def forward(self, features, pos_enc):
        # features, pos_enc: (B, N, D)
        
        # Attention
        q = (features + pos_enc).transpose(0, 1)  # Add positional encoding only to q and k
        k = (features + pos_enc).transpose(0, 1)  
        v = features.transpose(0, 1)
        attn_output, _ = self.self_attn(q, k, v) # (N, B, D)
        attn_output = features +  attn_output.transpose(0, 1) # (B, N ,D)
        attn_output = self.norm_attn(attn_output)

        # MLP
        ff = self.linear2(self.dropout(torch.relu(self.linear1(attn_output)))) # FC1 -> ReLU -> Dropout -> FC2 
        encoder_output = self.norm_mlp(attn_output + ff)
        return encoder_output



class Encoder(nn.Module):
    def __init__(self,
                 model_dim=model_config['model_dim'],
                 encoder_heads=model_config['encoder_heads'],
                 encoder_layers=model_config['encoder_layers']):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(model_dim, encoder_heads) for _ in range(encoder_layers)])

    def forward(self, features, pos_enc):
        for layer in self.layers:
            src = layer(features, pos_enc) # The positional encoder is added before every attention layer.
        return src
    

class DecoderLayer(nn.Module):
    def __init__(self,
                 model_dim=model_config['model_dim'],
                 decoder_heads=model_config['decoder_heads'],
                 mlp_inner_dim=model_config['mlp_inner_dim']):
        super().__init__()

        # Self Attention
        self.self_attn = nn.MultiheadAttention(model_dim, decoder_heads)
        self.norm_self_attn = nn.LayerNorm(model_dim)
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(model_dim, decoder_heads)
        self.norm_cross_attn = nn.LayerNorm(model_dim)
        
        # MLP
        self.linear1 = nn.Linear(model_dim, mlp_inner_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(mlp_inner_dim, model_dim)
        self.norm_mlp = nn.LayerNorm(model_dim)

    def forward(self, tgt, memory, pos_enc, query_pos):
        # Self-attention
        q = (tgt + query_pos).transpose(0, 1) # query positional encoding is only added to q and k
        k = (tgt + query_pos).transpose(0, 1) # (N, B, D)
        v = tgt.transpose(0, 1) 
        self_attn_output, _ = self.self_attn(q, k, v) # (N, B, D)
        tgt = self.norm_self_attn(tgt + self_attn_output.transpose(0, 1)) # (B, N ,D)

        # Cross-attention
        q = (tgt + query_pos).transpose(0, 1)         # the query positional encoding is added to object queries
        k = (memory + pos_enc).transpose(0, 1) # the positional encoding is added to the encoder's output
        v = memory.transpose(0, 1) # (N, B ,D)
        cross_attn_output, cross_attn = self.cross_attn(q, k, v) # (N, B ,D)
        tgt = self.norm_cross_attn(tgt + cross_attn_output.transpose(0, 1)) # (B, N ,D)

        # MLP
        ff = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = self.norm_mlp(tgt + ff) # (B, N, D)
        return tgt, cross_attn


class Decoder(nn.Module):
    def __init__(self,
                 model_dim=model_config['model_dim'],
                 decoder_heads=model_config['decoder_heads'],
                 decoder_layers=model_config['decoder_layers']):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(model_dim, decoder_heads) for _ in range(decoder_layers)])

        self.output_norm = nn.LayerNorm(model_dim)
        
    def forward(self, tgt, memory, pos_enc, query_pos):
        outputs = []
        cross_attn_weights = []
        for layer in self.layers:
            tgt, decoder_cross_attn = layer(tgt, memory, pos_enc, query_pos)
            cross_attn_weights.append(decoder_cross_attn)
            outputs.append(self.output_norm(tgt))

        output = torch.stack(outputs)    
        return output, torch.stack(cross_attn_weights)
    


class DETR(nn.Module):
    def __init__(self,
                 model_dim=model_config['model_dim'],
                 num_queries=model_config['num_queries'],
                 num_classes=dataset_config['num_classes'],
                # bakbone_model=model_config['backbone_model'],
                 max_len=5000):
        super().__init__()

        self.backbone = Backbone()

        self.encoder = Encoder()
        self.decoder = Decoder()

        # Learnable positional encoding for feature embeddings and object queries
        self.pos_enc = PositionalEncoding(max_len, model_dim)  # Learnable positional encoding

        self.query_embed = nn.Embedding(num_queries, model_dim)  # Learnable queries

        # Prediction
        self.class_mlp = nn.Linear(model_dim, num_classes)
        self.bbox_mlp = nn.Sequential(nn.Linear(model_dim, model_dim),
                                        nn.ReLU(),
                                        nn.Linear(model_dim, model_dim),
                                        nn.ReLU(),
                                        nn.Linear(model_dim, 4))

    def forward(self, x):
        # x: (B, 3, H, W)
        B = x.size(0)

        # Backbone
        features = self.backbone(x)                 # (B, N, D)

        pos_enc = self.pos_enc(features)            # (B, N, D) 

        # Encoder
        memory = self.encoder(features, pos_enc)    # (B, N, D)

        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, D)
        tgt = torch.zeros_like(query_pos)

        # Decoder
        hs, _ = self.decoder(tgt, memory, pos_enc, query_pos)  # (num_decoders, B, num_queries, D)

        # Prediction
        pred_logits = self.class_mlp(hs)  # (num_decoders, B, num_queries, num_classes)
        pred_bboxes = self.bbox_mlp(hs).sigmoid()  # (num_decoders, B, num_queries, 4)

        return {'pred_logits': pred_logits, 'pred_boxes': pred_bboxes}
