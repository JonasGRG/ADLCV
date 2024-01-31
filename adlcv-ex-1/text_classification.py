import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
import argparse

from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    print('training: ', len(train), 'test: ', len(test))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits((train, test), 
                                                       batch_size=batch_size, 
                                                       device=to_device()
    )

    return train_iter, test_iter


def main(args):
    # Set up the data iterator with arguments
    train_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                              batch_size=args.batch_size)

    # Create the model with specified arguments
    model = TransformerClassifier(embed_dim=args.embed_dim, 
                                  num_heads=args.num_heads, 
                                  num_layers=args.num_layers,
                                  pos_enc=args.pos_enc,
                                  pool=args.pool,  
                                  dropout=args.dropout,
                                  fc_dim=args.fc_dim,
                                  max_seq_len=MAX_SEQ_LEN, 
                                  num_tokens=VOCAB_SIZE, 
                                  num_classes=NUM_CLS)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')

    # Setup optimizer and scheduler with arguments
    opt = torch.optim.AdamW(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / args.warmup_steps, 1.0))

    # Define loss function
    loss_function = nn.CrossEntropyLoss()

    # Training loop
    for e in range(args.num_epochs):
        print(f'\n epoch {e}')
        model.train()
        for batch in tqdm.tqdm(train_iter):
            opt.zero_grad()
            input_seq = batch.text[0]
            batch_size, seq_len = input_seq.size()
            label = batch.label - 1
            if seq_len > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            out = model(input_seq)
            loss = loss_function(out, label) # compute loss
            loss.backward() # backward pass
            if args.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            opt.step()
            sch.step()

        # Validation loop
        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for batch in test_iter:
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                out = model(input_seq).argmax(dim=1)
                tot += float(input_seq.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            print(f'-- {"validation"} accuracy {acc:.3}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Classifier Parameters')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in the multi-head attention')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--pos_enc', type=str, default='fixed', choices=['fixed', 'learnable'], help='Type of positional encoding')
    parser.add_argument('--pool', type=str, default='max', choices=['max', 'mean', 'cls'], help='Pooling strategy')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--fc_dim', type=int, default=None, help='Fully connected layer dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=625, help='Warmup steps for learning rate scheduling')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--gradient_clipping', type=float, default=1, help='Gradient clipping threshold')

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main(args)
