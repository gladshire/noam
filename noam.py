import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab, build_vocab_from_iterator
import torch.nn as nn
import pandas as pd

import matplotlib.pyplot as plt

import string


__DEBUG__ = False


EMBEDDING_DIM = 1
HIDDEN_DIM = 64
NUM_LAYERS = 4
BATCH_SIZE = 32
DEVICE = "cpu"
NUM_EPOCHS = 50
NUM_WORKERS = 4
LEARNING_RATE = 0.001

SAMPLE_RATE = 10

# Define PyTorch Dataset wrapper for CSV
class EssaysDataset(Dataset):
    def __init__(self, csv_path, min_freq = 1, reserved_tokens = []):
        # Read CSV into a table
        self.data = pd.read_csv(csv_path)
	
        # Drop empty rows
        drop_rows = self.data[self.data['text']==''].index
        self.data.drop(drop_rows, inplace=True)

        # Define subset of data for debugging
        if __DEBUG__:
            self.data.sample(frac=1)
            self.data = self.data[:2500]

        # Define tokenizer and counter
        self.tokenizer = get_tokenizer("basic_english")
        self.counter = Counter()

        # Obtain tokens and update counter according to data
        for text_sample in self.data['text']:
            tokens = self.tokenizer(text_sample)
            self.counter.update(tokens)
        
        # Build vocabulary from tokens
        self.vocab = build_vocab_from_iterator([self.counter])
        self.max_seq_length = max(len(self.tokenizer(text_sample)) for text_sample in self.data['text'])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_sample = self.data.iloc[idx]['text']
        text_processed = self.preprocess(text_sample)
        text_sample_vector, seq_length = self.vectorize(text_processed)
        label = torch.tensor(self.data.iloc[idx]['generated'], dtype = torch.float32)
        return text_sample_vector, seq_length, label

    def preprocess(self, text):
        text_lower = text.lower()
        text_nopunc = text_lower.translate(str.maketrans('', '', string.punctuation))
        return text_nopunc

    def vectorize(self, text_sample):
        tokens = self.tokenizer(text_sample)
        if not tokens:
            tok_length = 1
            tokens = ['<unk>']
        else:
            tok_length = len(tokens)
        stoi_mapping = self.vocab.get_stoi()
        indices = [stoi_mapping.get(token, 0) for token in tokens]
        seq_length = len(indices)
        indices += [0] * (self.max_seq_length - seq_length)
        return torch.tensor(indices, dtype = torch.long), tok_length

# Define PyTorch model
class noamModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers = 1):
        super().__init__()

        '''
        torch.Size([32, 1502])
        torch.Size([32, 1502, 1])
        torch.Size([32, 1, 1502])
        torch.Size([32, 16, 165])
        torch.Size([32, 165, 16])
        torch.Size([32, 165, 128])
        torch.Size([32, 128, 165])
        torch.Size([32, 21120])
        '''
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn1 = nn.Sequential(
                        nn.LSTM(input_size=16, hidden_size=hidden_dim,
                                num_layers=2, bidirectional=True, batch_first=True)
        )

        self.cnn1 = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=4, kernel_size=8),
                        nn.MaxPool1d(3),
                        nn.ReLU(),

                        nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3),
                        nn.MaxPool1d(3),
                        nn.ReLU()
        )

        

        self.fc = nn.Sequential(
                        nn.Linear(25088, 512),
                        nn.ReLU(),

                        nn.Linear(512, 128),
                        nn.ReLU(),

                        nn.Linear(128, 32),
                        nn.ReLU(),

                        nn.Linear(32, 1),
        )


    def forward(self, x, lengths):

        y = self.embedding(x)

        y = y.permute(0, 2, 1)

        y = self.cnn1(y)

        y = y.permute(0, 2, 1)

        y, _ = self.rnn1(y)

        y = y.permute(0, 2, 1)

        y = y.reshape(y.size()[0], -1)

        output = self.fc(y)
        return torch.sigmoid(output)

# Initialize dataset
print("Reading/preparing training data ...")
dataset = EssaysDataset('Training_Essay_Data.csv')

vocab_size = len(dataset.vocab)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                            [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = False)


# Initialize model
print("Instantiating model ...")
model = noamModel(vocab_size, embedding_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, num_layers = NUM_LAYERS)
model.to(DEVICE)

# Define loss function, optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


ct = []
train_loss_hist = []

batch_num = 0
# Training loop
for epoch in range(NUM_EPOCHS):

    print("\n====================")
    print(f"Epoch: {epoch+1} of {NUM_EPOCHS}")
    print("====================")
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_false_pos = 0
    total_false_neg = 0

    for batch_idx, (text, lengths, labels) in enumerate(train_loader):
        text, lengths, labels = text.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(text, lengths)

        # Calculate loss
        loss = criterion(outputs.squeeze(), labels)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        # Update total loss
        total_loss += loss.item()

        # Obtain batch accuracy
        pred = torch.round(outputs).squeeze()
        correct = (pred == labels).sum().item()
        false_pos = ((pred == 1) & (labels == 0)).sum().item()
        false_neg = ((pred == 0) & (labels == 1)).sum().item()

        total_correct += correct
        total_samples += labels.size(0)
        total_false_pos += false_pos
        total_false_neg += false_neg

        if (batch_idx + 1) % SAMPLE_RATE == 0:
            batch_loss = loss.item()
            batch_acc = correct / BATCH_SIZE
            batch_t1e = false_pos / BATCH_SIZE
            batch_t2e = false_neg / BATCH_SIZE
            print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.4f}, E1: {batch_t1e:.4f}, E2: {batch_t2e:.4f}")

            batch_num += SAMPLE_RATE
            ct.append(batch_num)
            train_loss_hist.append(batch_loss)


    model.eval()
    total_loss_test = 0
    total_correct_test = 0
    total_samples_test = 0
    total_false_pos_test = 0
    total_false_neg_test = 0
    with torch.no_grad():
        for text, lengths, labels in test_loader:
            text, lengths, labels = text.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)

            outputs = model(text, lengths)

            test_loss = criterion(outputs.squeeze(), labels)

            total_loss_test += test_loss.item()

            pred = torch.round(outputs).squeeze()
            correct = (pred == labels).sum().item()

            false_pos_test = ((pred == 1) & (labels == 0)).sum().item()
            false_neg_test = ((pred == 0) & (labels == 1)).sum().item()

            total_correct_test += correct
            total_samples_test += labels.size(0)
            total_false_pos_test += false_pos_test
            total_false_neg_test += false_neg_test

    test_acc = total_correct_test / total_samples_test
    acc = total_correct_test / total_samples_test

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
    print(f"  Average training loss: {total_loss / len(train_loader):.4f}")
    print(f"  Average testing loss: {total_loss_test / len(test_loader):.4f}")
    print()
    print(f"  Average Training Accuracy: {acc:.4f}")
    print(f"    E1: {total_false_pos / len(train_loader):.4f}")
    print(f"    E2: {total_false_neg / len(train_loader):.4f}")
    print()
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"    E1: {total_false_pos_test / len(test_loader):.4f}")
    print(f"    E2: {total_false_neg_test / len(test_loader):.4f}")
    print()

    plt.plot(ct, train_loss_hist, color='blue')
    plt.title("BCE Loss")
    plt.xlabel("Batches")
    plt.grid()

    plt.savefig(f"train_{epoch+1}.png")
    plt.clf()
    plt.cla()
    if test_acc > 0.8:
        torch.save(model.state_dict(), "model.pt")
