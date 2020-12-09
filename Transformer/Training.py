class MovieReviewClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = Transformer(self.config)
        self.projection = nn.Linear(self.config.d_hidn, self.config.n_output, bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        # Activate Transformer class (Encoder Input + Decoder Input)
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, dec_inputs)
       
        # Max of Transformer model Output
        dec_outputs, _ = torch.max(dec_outputs, dim=1)
        
        # Activate Linear
        logits = self.projection(dec_outputs)
        
        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

import time
from tqdm import tqdm
import json

class MovieReviewDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels = []
        self.sentences = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
                data = json.loads(line)
                
                # Read labels from input
                self.labels.append(data["label"])
                
                # Read 'doc' token from input -> token id
                self.sentences.append([vocab.piece_to_id(p) for p in data["doc"]])
    
    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        return len(self.labels)
    
    def __getitem__(self, item):
        return (torch.tensor(self.labels[item]),
                torch.tensor(self.sentences[item]),
                #Decoder Input = [BOS]
                torch.tensor([self.vocab.piece_to_id("[BOS]")]))

    def movie_collate_fn(inputs):
    labels, enc_inputs, dec_inputs = list(zip(*inputs))

    # Add padding(0) to short sentences that Encoder Inputs length is equal
    # padding -> Vocab '-pad_id=0"
    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    
    # Add padding(0) to short sentences that Decoder Inputs length is equal
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [
        # Label-> tensor (by stack function)
        torch.stack(labels, dim=0),
        enc_inputs,
        dec_inputs,
    ]
    return batch

batch_size = 128
train_dataset = MovieReviewDataSet(vocab, "./ratings_train.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=movie_collate_fn)
test_dataset = MovieReviewDataSet(vocab, "./ratings_test.json")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=movie_collate_fn)


def evaluate_epoch(config, model, data_loader):
    matchs = []
    model.eval()

    n_word_total = 0
    n_correct_total = 0
    with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
        for i, value in enumerate(data_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            # Activate MovieReviewClassification (Encoder input + Decoder input)
            outputs = model(enc_inputs, dec_inputs)
            
            # first output -> Prediction logits
            logits = outputs[0]
            
            # Max logits index
            _, indices = logits.max(1)

            # Compare Max logits index and labels (torch.eq function)
            match = torch.eq(indices, labels).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

def train_epoch(config, epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            
            # Activate MovieReviewClassification
            outputs = model(enc_inputs, dec_inputs)
            
            #first output -> Prediction logits
            logits = outputs[0]

            # Calculate Loss (using logits, labels)
            loss = criterion(logits, labels)
            loss_val = loss.item()
            losses.append(loss_val)

            # Train
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

# Check use of GPU 
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Negative(0), Positive(1) -> 2 Outputs
config.n_output = 2
print(config)

learning_rate = 5e-5

# 10 epoch
n_epoch = 10

# MovieReviewClassification
model = MovieReviewClassification(config)
model.to(config.device)

# loss function
criterion = torch.nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses, scores = [], []
for epoch in range(n_epoch):
    # Train for each epoch
    loss = train_epoch(config, epoch, model, criterion, optimizer, train_loader)
    
    # Evaluate for each epoch
    score = evaluate_epoch(config, model, test_loader)

    losses.append(loss)
    scores.append(score)

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "loss": losses,
    "score": scores
}
df = pd.DataFrame(data)
display(df)

# graph
plt.figure(figsize=[12, 4])
plt.title('loss and score')
plt.plot(losses, label="loss")
plt.plot(scores, label="score")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()