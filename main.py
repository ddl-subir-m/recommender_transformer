from tokenizer import Tokenizer
from train import train_model

def main():
    texts = [
        "user1 viewed item1 item2 item3",
        "user2 viewed item2 item4 item5 item1",
        "user3 viewed item3 item1",
    ]

    vocab_size = 100
    tokenizer = Tokenizer(texts, vocab_size)

    model_params = {
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'max_seq_len': 10,
        'dropout': 0.1
    }

    train_params = {
        'num_epochs': 10,
        'learning_rate': 0.001
    }

    model = train_model(texts, tokenizer, model_params, train_params)

    # Inference example
    device = next(model.parameters()).device
    test_sequence = torch.tensor(tokenizer.encode("user4 viewed item2 item5")[:model_params['max_seq_len']]).unsqueeze(0).to(device)
    test_mask = (test_sequence != 0).unsqueeze(1).unsqueeze(2).to(device)
    
    model.eval()
    with torch.no_grad():
        predicted_item_index = model.predict_next_item(test_sequence, test_mask)
    predicted_item = tokenizer.decode([predicted_item_index.item()])
    print(f"Predicted next item: {predicted_item}")

if __name__ == "__main__":
    main()