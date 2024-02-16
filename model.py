import torch
import torch.nn as nn


class SpamClassifier(nn.Module):
  def __init__(self, vocab_size, max_seq_len, d_model, n_encoders, n_heads, ff_dim, dropout):
    #batch, seq_len
    #batch, seq_len, d_model
    super().__init__()
    self.max_seq_len= max_seq_len

    self.embeddings = nn.Embedding(vocab_size, d_model)
    self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first = True)
    self.encoder = nn.TransformerEncoder(encoder_layer, n_encoders)

    self.linear = nn.Linear(d_model, 2)

  def forward(self,x, mask):
    batch_size = x.shape[0]
    x = self.embeddings(x)
    pe = self.pos_embedding(torch.arange(0, self.max_seq_len).unsqueeze(0).repeat(batch_size, 1 ).to(x.device))
    assert x.shape == pe.shape, 'embedding shape mismatch'
    x = x+ pe
    x = self.encoder(src= x, src_key_padding_mask  = mask)

    x = self.linear(x[:,0,:])

    return x