import torch
import torch.nn as nn
from model import TransformerRecommender
from utils import preprocess_data

def train_model(texts, tokenizer, model_params, train_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_data, input_masks = preprocess_data(texts, tokenizer, model_params['max_seq_len'])
    input_data = input_data.to(device)
    input_masks = input_masks.to(device)

    model = TransformerRecommender(
        len(tokenizer.word_to_index),
        model_params['d_model'],
        model_params['num_heads'],
        model_params['num_layers'],
        model_params['d_ff'],
        model_params['max_seq_len'],
        model_params['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])

    model.train()
    for epoch in range(train_params['num_epochs']):
        optimizer.zero_grad()
        output = model(input_data, input_masks)
        loss = criterion(output[:, :-1, :].contiguous().view(-1, len(tokenizer.word_to_index)), input_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model