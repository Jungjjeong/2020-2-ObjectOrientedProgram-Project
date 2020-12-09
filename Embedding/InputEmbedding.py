# vocab count
n_vocab = len(vocab) 

# hidden size
d_hidn = 128 

# embedding object
nn_emb = nn.Embedding(n_vocab, d_hidn) 

# input embedding
input_embs = nn_emb(inputs) 
print(input_embs.size())