import math
import torch
import torch.nn as nn
import numpy as np

 
#! Embedding Layer
 #* The usual flow is (sentence) -> (tokenize) -> (word_to_index) -> (index_to_embedding) -> (embedding_matrix)
 
# d_model is the most important number in the model, it is the number of dimensions in the embedding vector
#^ Usually d_model = 512
# vocab_size is the number of words in the vocabulary
 
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)   # This is the embedding matrix, a lookup table that maps word indices to word embeddings
                                                             # Creates an embedding matrix of shape (vocab_size, d_model)   
                                                             # When you pass a word index (42), it fetches the corresponding embedding vector from this matrix.
    
    def forward(self, x):
        #^ (batch, sent_length) -> (batch, sent_length, d_model)
        return self.embedding(x) * np.sqrt(self.d_model)     # Multiply by sqrt(d_model) to scale the embeddings, as per the paper
    


#! Positional Encoding
 #* Positional encoding provides information about word positions in a sentence.
# It is added to the input embeddings element-wise.
#* The positional embedding is computed only once and reused for every sentece during training and inference. They arent trainable.
# Each value inside the d_model vector (of 512) corresponds to a specific feature or dimension, i.

#^ Create a matrix of shape (sent_length, d_model) where each row is a positional encoding vector
#^ (batch, sent_length, d_model) -> (batch, sent_length, d_model) + (1, sent_length, d_model) -> (batch, sent_length, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):  # seq_len is the maximum length of a sentence, not the length of the input sentence, say 5000
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a positional encoding matrix
        pe = torch.zeros(seq_len, d_model)     #^ like (5000,512)
        
        # now we need the position from [1:5000] with shape (5000,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )                                                                   # Compute the positional encodings once in log space, dont think too much about this!
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)                                                #^(1, sent_length, d_model)
        
        #register the positional encoding matrix as a buffer so that it is saved along with the model
        self.register_buffer('pe', pe)       #This is a buffer, it is not a parameter of the model, but it is part of the model state
                                             #It does not get updated during training (not a learnable parameter)
                                             
    def forward(self, x):
        with torch.no_grad():
            pos = self.pe[:, :x.size(1), :]            # dont update the positional encoding
            
        x = x + pos                                  
                                                                            # meaning add all the batch, only till the sentence length, all the d_model
        return self.dropout(x)                                              # Apply dropout to the positional encoding
    
    
    
    
#! Layer Normalization 
 #^ (batch, sent_length, d_model) -> (batch, sent_length, d_model)
 
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 10**-6):       
         super().__init__()
         self.eps = eps                                   # eps is a small number to avoid division by zero
         self.d_model = d_model
         self.alpha = nn.Parameter(torch.ones(d_model))   # Learnable parameter of shape (d_model,)
         self.bias = nn.Parameter(torch.zeros(d_model))   # Learnable parameter of shape (d_model,)
         
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)             # dim=-1 means taking the mean across the last dimension, which is hidden_size
        std = x.std(dim=-1, keepdim = True)               # so it takes the mean and std for each word in the sentence. Calc the mean and std of 512 values
        #^ both mean and std are of shape (batch, sent_length, 1) because we used keepdim=True
        
        return (((x-mean)/(std + self.eps)) * self.alpha) + self.bias
  
  
    
#! Feed Forward Layer
#^ (batch, sent_length, d_model) -> (batch, sent_length, d_model)
 
class FeedForwardBlock():
    def __init__(self, d_model: int, dff: int, dropout: float):
        self.d_model = d_model
        self.dff = dff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dff)          # Linear layer 1 - Has W1 and b1
        self.linear2 = nn.Linear(dff, d_model)          # Linear layer 2 - Has W2 and b2
        self.relu = nn.ReLU()
    
    def forward(self, x):
        #^ (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    

#! Transformer Block
 #^ (batch, sent_length, d_model) -> 3 * (batch, sent_length, d_model) * (d_model, d_model) -> 3 * (batch, sent_length, d_model) -> (batch, sent_length, h, d_k)
 #^ h * d_k = d_model             h = num_heads
 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)
        
        self.w_q = nn.Linear(d_model, d_model)   # Linear layer for the Query      #you can also initalize the w_q to torch.ones and manually get the weights by w_q @ x
        self.w_k = nn.Linear(d_model, d_model)   # Linear layer for the Key
        self.w_v = nn.Linear(d_model, d_model)   # Linear layer for the Value
        
        self.w_o = nn.Linear(d_model, d_model)   # Linear layer for the output
        
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        eps = 1e-9
        
        num = query @ key.transpose(-2, -1)                                    #^ (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        denom = math.sqrt(d_k + eps)                                           #^ (batch, h, seq_len, seq_len)
        frac = num / denom
        
        #But before softmax, we need to apply the mask
        if mask is not None:
            frac.masked_fill_(mask == 0, -1e9)                                 #^ (batch, h, seq_len, seq_len)
        # So wherever the mask is 0, we replace that value with -1e9 in frac
        
        #now apply softmax
        softmax = nn.Softmax(dim=-1)
        attent = softmax(frac)                                              #^ (batch, h, seq_len, seq_len)
        if dropout is not None:
            attent = dropout(attent)
        
        return (attent @ value), frac                                       #^ (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)                      #^ (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)                        
        value = self.w_v(v)
        
        #^ (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        query = query.view(query.size(0), query.size(1), self.h, self.d_k)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k)
        
        #^ (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.transpose(1, 2)                                           # Swaps 2nd and 3rd dimensions
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        attent, frac = MultiHeadAttention.attention(query, key, value, mask, self.dropout)  #^ (batch, h, seq_len, d_k) 
        
        #Now concat all the heads
        #^ (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k)
        attent = attent.transpose(1, 2)
        
        #contiguous() will rearrange the memory allocation so that the tensor is contiguous
        # This is necessary for the view operation below
        #^ (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        attent = attent.contiguous().view(attent.size(0), -1, self.d_model)
        
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(attent)
    
    
    
#! Residual Connection
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:         # features = d_model
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
        
    
#! Encodere Block
# Now the init will take all the arguments in the init of the enclosing layers

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:    # features = d_model
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        
        #We will also use `nn.ModuleList` which will hold all the submodles in a list. You can have many modules inside and can index them
        self.sublayers = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) #As we have 2 residual connections
        
        
    def forward(self, x, src_mask):
        # So first we pass the input through the first sublayer that is the self attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        #We use this lamda anol cause we dont want to rewrite the x, as it is the same x that is passed to the self attention block and the residual connection
        
        # Then we pass the output of the first sublayer through the second sublayer that is the feed forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
        
    

#! Encoder
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:    # Layers is a list of EncoderBlocks, so we will call the forward method of each block when we pass x
        super().__init__()
        self.layers = layers
        # Add layernorm at the end of the encoder
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)               #Call the forward method of each encoder block
        return self.norm(x)                  #The x has passed through all the N blocks, now we apply layer norm
    
    #* Number of encoder blocks = len(self.layers)
    
    

#! Decoder Block
class DecoderBlock(nn.Module):
    
    # Useless to use module list for attention, as we have self and cross attention and list wouuld be unnecessary
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, 
                cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, 
                dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block 
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)]) # 3 residual connections
        
    #x is the imput of the decoder, encoder_output is the output of the encoder, src_mask is the mask for the encoder output
    
    #We will have 2 masks, one for the self attention block and one for the cross attention block
    #The self attention block will have a mask for the target sentence, and the cross attention block will have a mask for the source sentence
    #src_mask is the mask for the source sentence, tgt_mask is the mask for the target sentence
    
    # `src_mask` → Mask for the encoder output (used in the encoder-decoder attention layer)
    # `tgt_mask` → Mask for the decoder input (used in the self-attention layer of the decoder)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # residual connection 1 itself has a layer norm layer inside it
        
        #Now the query comes from the decoder, and the key and value come from the encoder.  So (x, encoder_output, encoder_output)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        #Now feed forward block
        x= self.residual_connections[2](x, self.feed_forward_block)
        
        return x
    
    
    
#! Decoder 
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        # Add layernorm at the end of the decoder
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

#! Projection Layer
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    
    

#! Transformer
# encoder: An instance of the Encoder that processes the source sequence.
# decoder: An instance of the Decoder that generates the target sequence based on the encoder output.
# src_embed: An instance of InputEmbeddings that converts source tokens into dense vectors (d_model dimensions).
# tgt_embed: An instance of InputEmbeddings that converts target tokens into dense vectors (d_model dimensions).
# src_pos: An instance of PositionalEncoding that adds positional information to the source embeddings.
# tgt_pos: An instance of PositionalEncoding that adds positional information to the target embeddings.
# projection_layer: A linear layer that maps the decoder output to vocabulary logits (used for predicting words).
# src: The source sequence (batch of token indices) passed as input.
# src_mask: A mask for the source sequence to ignore padding tokens.
# encoder_output: The encoded representation of the source sequence (from encode).
# src_mask: The mask for the source sequence (same as in encode).
# tgt: The target sequence (batch of token indices) passed as input.
# tgt_mask: A mask for the target sequence to prevent attending to future tokens (causal mask).


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        #^ (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        #^ (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        #^ (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    
    
#! Transformer Initialization
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, 
                      tgt_seq_len: int, d_model: int=512, 
                      N: int=6, h: int=8, 
                      dropout: float=0.1, 
                      d_ff: int=2048) -> Transformer:
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    #Create N blocks of the encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    
    # Initialize the parameters
    #! We use xavier uniform initialization for the weights and zeros for the biases. We use this instead of random weights
    # Random weights will have higher variance and make the gradients explode or vanish. So we use xavier uniform initialization, which scales down the weights
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
