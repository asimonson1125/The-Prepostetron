"""
I asked chatGPT to do this project for me.  We went back and forth for a while and ended up with this.  Here's what it calls 'the Pepepoktron'
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.bn1(self.fc3(x))
        return x

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Function to load patent title-description pairs from your dataset
def load_data(file_path, nrows=None):
    # Load your data from the file_path and preprocess it if necessary
    # For example, you can use pandas to load a CSV file containing your data
    # You may need to adjust this based on the format of your data
    data = None
    if nrows == None:
        data = pd.read_csv(file_path, sep='\t')
    else:
        data = pd.read_csv(file_path, sep='\t', nrows=nrows)
    titles = data['patent_title'].values
    descriptions = data['patent_abstract'].values
    return encode_text(titles, descriptions)


tokenizer = get_tokenizer("basic_english")
def encode_text(titles, descriptions):
    # Tokenization
    titles_tokens = [tokenizer(title) for title in titles]
    descriptions_tokens = [tokenizer(desc) for desc in descriptions]

    # Vocabulary building
    global vocab
    vocab = build_vocab_from_iterator(['<%$PADDING$%>'] + titles_tokens + descriptions_tokens)
    # Added padding keyword to remove extra 'words' from short titles (see idx != 0 in decoder)

    # Numerical encoding
    titles_encoded = [[vocab[token] for token in title_tokens] for title_tokens in titles_tokens]
    descriptions_encoded = [[vocab[token] for token in desc_tokens] for desc_tokens in descriptions_tokens]
    
    # Padding (assuming maximum sequence length)
    max_title_length = len(max(titles_encoded, key=len))
    max_desc_length = len(max(descriptions_encoded, key=len))
    titles_padded = [title[:max_title_length] + [0] * (max_title_length - len(title)) for title in titles_encoded]
    descriptions_padded = [desc[:max_desc_length] + [0] * (max_desc_length - len(desc)) for desc in descriptions_encoded]

    return titles_padded, descriptions_padded

def decode_output(output):

    # Map numerical indices back to tokens using the vocabulary
    decoded_tokens = [vocab.vocab.itos_[idx] for idx in output if idx != 0] # idx == 0 is padding

    # Concatenate the tokens to form the generated text
    generated_text = ' '.join(decoded_tokens)

    return generated_text


# Function to generate patent title-description pairs for training
def generate_data_batch(titles, descriptions, batch_size):
    # Generate a random batch of title-description pairs from your loaded data
    indices = np.random.choice(len(titles), batch_size, replace=True)
    batch_titles = torch.tensor([titles[idx] for idx in indices])
    batch_descriptions = torch.tensor([descriptions[idx] for idx in indices])
    return batch_titles, batch_descriptions

# Function to train the GAN
def train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, num_epochs, batch_size, titles, descriptions):
    generator.train()
    discriminator.train()
    total_batches = num_epochs * num_batches
    with tqdm(total=total_batches) as pbar:
        for epoch in range(num_epochs):
            for batch_idx in range(num_batches):
                # Generate a batch of title-description pairs
                batch_titles, batch_descriptions = generate_data_batch(titles, descriptions, batch_size)

                # Train discriminator
                d_optimizer.zero_grad()
                d_real_output = discriminator(torch.cat((batch_titles, batch_descriptions), dim=1))
                d_real_loss = criterion(d_real_output, torch.ones(batch_size, 1))
                
                fake_titles = generator(batch_descriptions)
                d_fake_output = discriminator(torch.cat((fake_titles, batch_descriptions), dim=1))
                d_fake_loss = criterion(d_fake_output, torch.zeros(batch_size, 1))

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()

                # Train generator
                g_optimizer.zero_grad()
                fake_titles = generator(batch_descriptions)
                g_output = discriminator(torch.cat((fake_titles, batch_descriptions), dim=1))
                g_loss = criterion(g_output, torch.ones(batch_size, 1))
                g_loss.backward()
                g_optimizer.step()
                
                if batch_idx % 10 == 0 or batch_idx + 1 == num_batches:
                    dloss = d_loss.item()
                    gloss = g_loss.item()
                pbar_str = f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{num_batches}] '
                pbar_str += ' ' * (33 - len(pbar_str))
                loss_str = f'D Loss: {dloss:0.2f}, G Loss: {gloss:0.2f}'
                loss_str += ' ' * (30 - len(loss_str))
                pbar_str += loss_str
                pbar.update(1)
                pbar.set_description(pbar_str)
                
def generate_patent(generator, description):
    generator.eval()
    with torch.no_grad():
        # Convert the description to a tensor
        description_tensor = torch.tensor(description).unsqueeze(0)

        # Generate a patent title using the generator
        generated_title = generator(description_tensor)

        return generated_title

if __name__ == "__main__":
    # Load your actual patent title-description pairs from your dataset
    file_path = "datasets/g_patent/g_patent.tsv"
    print("Loading patents...")
    titles, descriptions = load_data(file_path, nrows=10000)
    vocabSize = len(vocab.vocab.itos_) # WIP
    # TODO:
    # Apply vocabsize to make output tokens
    print(f"Loaded {len(titles)} patents!")

    # Define hyperparameters
    title_dim = len(max(titles, key=len))
    desc_dim = len(max(descriptions, key=len)) 
    input_dim = title_dim + desc_dim
    batch_size = 64
    num_epochs = 10
    num_batches = len(titles) // batch_size

    # Initialize networks, optimizers, and loss criterion
    generator = Generator(desc_dim, title_dim)
    discriminator = Discriminator(input_dim)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    # Train the GAN
    train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, num_epochs, batch_size, titles, descriptions)
    
    sampleOutput = generate_patent(generator, descriptions[0])
    print(f"Real title: {decode_output(titles[0])}\nFake title: {decode_output(sampleOutput)}\n\nDescription:\n{decode_output(descriptions[0])}")
