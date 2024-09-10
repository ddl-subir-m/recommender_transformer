import torch
from torch.nn.utils.rnn import pad_sequence

def preprocess_data(texts, tokenizer, max_seq_len):
    encoded_texts = [torch.tensor(tokenizer.encode(text)[:max_seq_len]) for text in texts]
    padded_texts = pad_sequence(encoded_texts, batch_first=True, padding_value=0)
    masks = (padded_texts != 0).unsqueeze(1).unsqueeze(2)
    return padded_texts, masks