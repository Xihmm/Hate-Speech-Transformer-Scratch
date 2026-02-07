''' 
Encoder:
TransformerEncoder:
    PositionEncoding
    TransformerEncoderLayers
        SingleHeadAttention
        feedforward
        norm1
        feedforward
        norm2
    TransformerEncoderLayers
        SingleHeadAttention
        feedforward
        norm1
        feedforward
        norm2
    TransformerEncoderLayers
        SingleHeadAttention
        feedforward
        norm1
        feedforward
        norm2

'''
import torch
import torch.nn as nn
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(SingleHeadAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1) # dim=-1 means the last dimension
        self.output = nn.Linear(d_model, d_model) # output layer
    
    def forward(self, query, key, value, mask = None):
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model).float())
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = self.softmax(attention)
        context = torch.matmul(attention, V)
        output = self.output(context)
        return output, attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # Dimension of each head
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len, depth)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.query(query)  # Shape: (batch_size, seq_len, d_model)
        K = self.key(key)      # Shape: (batch_size, seq_len, d_model)
        V = self.value(value)  # Shape: (batch_size, seq_len, d_model)
        
        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # Shape: (batch_size, num_heads, seq_len, depth)
        K = self.split_heads(K, batch_size)  # Shape: (batch_size, num_heads, seq_len, depth)
        V = self.split_heads(V, batch_size)  # Shape: (batch_size, num_heads, seq_len, depth)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth).float())
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_output = torch.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, seq_len, depth)
        
        # Concatenate heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, seq_len, num_heads, depth)
        attention_output = attention_output.view(batch_size, -1, self.d_model)  # Shape: (batch_size, seq_len, d_model)
        
        # Final linear layer
        output = self.output(attention_output)  # Shape: (batch_size, seq_len, d_model)
        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # self.attention = SingleHeadAttention(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads=8)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output, attention = self.attention(x,x,x)
        x = self.norm1(x + self.dropout(output))
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.position_encoding = PositionEncoding(d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, dim_feedforward, dropout) for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=256, num_layers=6, dim_feedforward=512, max_len=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, dim_feedforward, dropout)
        self.position_encoding = PositionEncoding(d_model, max_len)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Step 1: Embed tokens and add positional encoding
        x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)
        x = self.position_encoding(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Step 2: Pass through the Transformer encoder
        x = self.encoder(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Step 3: Pooling (reduce sequence dimension)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, d_model, seq_len)
        x = self.pooling(x).squeeze(-1)  # Shape: (batch_size, d_model)
        
        # Step 4: Classification head
        logits = self.fc(x)  # Shape: (batch_size, num_classes)
        return logits
    

''' 
Input (Shifted Target Sequence)
        ↓
[Masked Self-Attention]
        ↓
[Add & Norm]
        ↓
[Cross-Attention (Encoder-Decoder)]
        ↓
[Add & Norm]
        ↓
[Feedforward Network]
        ↓
[Add & Norm]
'''
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.masked_self_attention = SingleHeadAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attention = SingleHeadAttention(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention
        attn_output, attn_weights_self = self.masked_self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Cross-Attention
        cross_output, attn_weights_cross = self.cross_attention(x, encoder_output, encoder_output, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_output))

        # Feedforward Network
        ff_output = self.feedforward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x, attn_weights_self, attn_weights_cross


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, dim_feedforward, dropout=0.1, max_len=512):
        super(TransformerDecoder, self).__init__()
        self.position_encoding = PositionEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        # Add positional encoding to target embeddings
        tgt = self.position_encoding(tgt)

        # Pass through each decoder layer
        for layer in self.layers:
            tgt = layer(tgt, encoder_output, tgt_mask, memory_mask)

        return self.norm(tgt)  # Normalize output for stability
class Transformer_Encoder_Decoder(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, dim_feedforward, vocab_size, dropout=0.1, max_len=512):
        super(Transformer_Encoder_Decoder, self).__init__()
        
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = TransformerEncoder(num_encoder_layers, d_model, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, dim_feedforward, dropout, max_len)

        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Pass source through the encoder
        encoder_output = self.encoder(src)

        # Pass target and encoder output through the decoder
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

        return decoder_output

class TransformerEncoderModel(nn.Module):
    def __init__(self, num_layers, d_model, dim_feedforward, vocab_size, num_classes=None, max_len=512, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        
        # Token Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer Encoder (Your Implementation)
        self.encoder = TransformerEncoder(num_layers, d_model, dim_feedforward, dropout)
        

    def forward(self, x):
        # Step 1: Token Embeddings
        x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)

        # Step 2: Pass through Encoder
        encoder_output = self.encoder(x)  # Shape: (batch_size, seq_len, d_model)
        
        return encoder_output
    