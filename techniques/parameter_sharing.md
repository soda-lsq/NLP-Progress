Share parameters between Embedding Layer and Output Layer, both have matrix size of voc_size * token_emb_dim (maybe 256), used in model definition part. 

```
 if config['share_inout_param']:
     self.out_bias = nn.Parameter(torch.zeros(self.voc_size).uniform_(0.01))
     self.emb2score = lambda x: F.linear(x, self.token_emb.weight, self.out_bias)
 else:
     self.emb2score = nn.Linear(self.token_emb_dim, self.voc_size)
```
