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

# Global settings/hyperparameters for NLP model
EMBEDDING_DIM = 1
HIDDEN_DIM = 256
NUM_LAYERS = 4
BATCH_SIZE = 64
DEVICE = "cpu"
NUM_EPOCHS = 25
NUM_WORKERS = 4
LEARNING_RATE = 0.001
MOMENTUM = 0.9

SAMPLE_RATE = 10 # For loss plotting, number of batches between points

# Utility function for plotting two series
def plotLists(x, y, color, xaxis, yaxis, title, filename):
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.grid()

    plt.savefig(filename)
    plt.clf()
    plt.cla()



# Define PyTorch Dataset wrapper for CSV of Essay data, formatted:
#
#   Essay	Generated
#   essay1...   0
#   essay2...   1
#   essay3...   1
#   ...   
class EssaysDataset(Dataset):
    def __init__(self, csv_path, min_freq = 1, reserved_tokens = [],
                 max_length = 0):
        # Read CSV into a table
        self.data = pd.read_csv(csv_path)
	
        # Drop empty rows
        drop_rows = self.data[self.data['text']==''].index
        self.data.drop(drop_rows, inplace=True)

        # Define tokenizer and counter
        self.tokenizer = get_tokenizer("basic_english")
        self.counter = Counter()

        # Obtain tokens and update counter according to data
        for text_sample in self.data['text']:
            tokens = self.tokenizer(text_sample)
            self.counter.update(tokens)
        
        # Build vocabulary from tokens
        self.vocab = build_vocab_from_iterator([self.counter])
        if max_length == 0:
            self.max_seq_length = max(len(self.tokenizer(text_sample)) for text_sample in self.data['text'])
        else:
            self.max_seq_length = max_length
        #print(self.max_seq_length)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_sample = self.data.iloc[idx]['text']
        text_processed = self.preprocess(text_sample)
        text_sample_vector, seq_length = self.vectorize(text_processed)
        label = torch.tensor(self.data.iloc[idx]['generated'], dtype = torch.float32)
        return text_sample_vector, seq_length, label

    # Preprocessing function for removing upper case, punctuation from essays
    def preprocess(self, text):
        text_lower = text.lower()
        text_nopunc = text_lower.translate(str.maketrans('', '', string.punctuation))
        return text_nopunc

    # Tokenizer/vectorizer function for essay data, reduces dimensionality
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

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Recurrent LSTM layer for capturing global sequential patterns
        self.rnn1 = nn.Sequential(
                        nn.LSTM(input_size=16, hidden_size=hidden_dim,
                                num_layers=2, bidirectional=True, batch_first=True)
        )

	# Define convolutional downsampling step
        self.cnn1 = nn.Sequential(
                        # First convolutional layer, kernel size of 12 for extracting
                        # word-sentence level features (in theory)
                        nn.Conv1d(in_channels=1, out_channels=4, kernel_size=12),
                        nn.MaxPool1d(8),
                        nn.BatchNorm1d(4),
                        nn.ReLU(),

			# Second convolutional layer, kernel size of 4 for extracting
                        # paragraph-level features from sentence-level ones (again, in theory)
                        nn.Conv1d(in_channels=4, out_channels=16, kernel_size=4),
                        nn.MaxPool1d(8),
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
        )

        # Transformer layer for self-attention. Not currently in-use
        self.tns = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=hidden_dim*2, nhead=2,
                                                   dim_feedforward=128, batch_first=True),
                        num_layers=6
        )

        # Fully connected layers to get an output
        self.fc = nn.Sequential(
                        nn.Linear(13824, 256),
                        nn.ReLU(),

                        nn.Linear(256, 64),
                        nn.ReLU(),

                        nn.Linear(64, 8),
                        nn.ReLU(),

                        nn.Linear(8, 1),
        )


    def forward(self, x, lengths):
        # Apply embedding to tokenized text
        y = self.embedding(x)

        y = y.permute(0, 2, 1)

        y = self.cnn1(y)

        y = y.permute(0, 2, 1)

        y, _ = self.rnn1(y)
 
        #y = self.tns(y)

        y = y.permute(0, 2, 1)

        y = y.reshape(y.size()[0], -1)

        output = self.fc(y)
        return torch.sigmoid(output)

# Initialize dataset
print("Reading/preparing training and testing data ...")

train_dataset = EssaysDataset('Training_Essay_Data.csv', max_length = 1780)
test_dataset = EssaysDataset('Test_Essay_Data.csv', max_length = 1780)

train_vocab_size = len(train_dataset.vocab)
test_vocab_size = len(test_dataset.vocab)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)



# Initialize model
print("Instantiating model ...")
model = noamModel(train_vocab_size, embedding_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, num_layers = NUM_LAYERS)
model.to(DEVICE)

# Define loss function, optimizer
criterion = nn.BCELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


ct = []
train_loss_hist = []
train_acc_hist = []
test_acc_hist = []

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

        # Obtain batch accuracy, type-1 and type-2 error rates
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
            train_acc_hist.append(batch_acc)

    # Plot training loss and accuracy
    plotLists(ct, train_loss_hist, 'blue', "Batches", "Loss", "BCE Loss", f"train_{epoch+1}.png")
    plotLists(ct, train_acc_hist, 'orange', "Batches", "% Correct", "Accuracy", f"train_acc_{epoch+1}.png")

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
    acc = total_correct / total_samples_test
    
    test_acc_hist.append(test_acc)
    if epoch > 0:
        plotLists(range(epoch+1), test_acc_hist, 'red', "Epochs", "% Correct", "Average Test Accuracy",
                  f"test_{epoch+1}.png")

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
    print(f"  Average training loss: {total_loss / total_samples:.4f}")
    print(f"  Average testing loss: {total_loss_test / total_samples_test:.4f}")
    print()
    print(f"  Average Training Accuracy: {acc:.4f}")
    print(f"    E1: {total_false_pos / total_samples:.4f}")
    print(f"    E2: {total_false_neg / total_samples:.4f}")
    print()
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"    E1: {total_false_pos_test / total_samples_test:.4f}")
    print(f"    E2: {total_false_neg_test / total_samples_test:.4f}")
    print()


    if test_acc > 0.8:
        torch.save(model.state_dict(), "model.pt")

