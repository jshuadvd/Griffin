import gzip
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import pytorch_lightning as pl
from griffin_torch import Griffin

# constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024
DATA_SIZE = 50 * 1024 * 1024  # 50MB

print(f"Script started")
print(f"Data size:", DATA_SIZE)


# helpers
def cycle(loader):
    while True:
        yield from loader


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(
            0, self.data.size(0) - self.seq_len, (1,)
        )
        full_seq = self.data[
            rand_start : rand_start + self.seq_len + 1
        ].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len


class GriffinLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Griffin(
            num_tokens=256,
            dim=512,
            depth=8,
            seq_len=SEQ_LEN,
            mlp_mult=4,
            heads=8,
            dropout=0.1,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_seq, target_seq = batch[:, :-1], batch[:, 1:]
        output = self.model(input_seq)
        loss = self.criterion(
            output.view(-1, output.size(-1)), target_seq.reshape(-1)
        )
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, target_seq = batch[:, :-1], batch[:, 1:]
        output = self.model(input_seq)
        loss = self.criterion(
            output.view(-1, output.size(-1)), target_seq.reshape(-1)
        )
        self.log(
            "val_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)

    def generate(self, inp, generate_length):
        # Assuming the 'generate' method is implemented correctly in the Griffin model
        return self.model.generate(inp, generate_length)


def main():
    print(f"Entering main function")

    # prepare enwik8 data
    with gzip.open("./data/enwik8.gz") as file:
        X = np.frombuffer(file.read(DATA_SIZE), dtype=np.uint8)
        train_size = int(len(X) * 0.9)
        trX, vaX = X[:train_size], X[train_size:]
        data_train, data_val = (
            torch.from_numpy(trX).clone(),
            torch.from_numpy(vaX).clone(),
        )

    print(f"Training data shape: {data_train.shape}")
    print(f"Validation data shape: {data_val.shape}")

    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=4
    )

    # instantiate model and trainer
    model = GriffinLightningModule()
    print(model)

    trainer = pl.Trainer(
        max_steps=NUM_BATCHES,
        val_check_interval=VALIDATE_EVERY,
        log_every_n_steps=10,
        accumulate_grad_batches=GRADIENT_ACCUMULATE_EVERY,
    )
    print(trainer)

    print("Starting training...")

    # training
    trainer.fit(model, train_loader, val_loader)

    # generation
    if trainer.global_step % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print("%s \n\n %s" % (prime, "*" * 100))

        sample = model


if __name__ == "__main__":
    main()

# import gzip
# import random

# import numpy as np
# import torch
# import tqdm
# from torch.utils.data import DataLoader, Dataset
# from torch.optim import AdamW

# from griffin_torch import Griffin

# # constants
# NUM_BATCHES = int(1e5)
# BATCH_SIZE = 4
# GRADIENT_ACCUMULATE_EVERY = 4
# LEARNING_RATE = 2e-4
# VALIDATE_EVERY = 100
# GENERATE_EVERY = 500
# GENERATE_LENGTH = 512
# SEQ_LEN = 1024

# # helpers
# def cycle(loader):
#     while True:
#         yield from loader

# def decode_token(token):
#     return str(chr(max(32, token)))

# def decode_tokens(tokens):
#     return "".join(list(map(decode_token, tokens)))

# # instantiate GPT-like decoder model
# model = Griffin(
#     num_tokens=256,
#     dim=512,
#     depth=8,
#     seq_len=SEQ_LEN,
#     mlp_mult=4,
#     heads=8,
#     dropout=0.1,
# )

# # prepare enwik8 data
# with gzip.open("./data/enwik8.gz") as file:
#     X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
#     trX, vaX = np.split(X, [int(90e6)])
#     data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

# class TextSamplerDataset(Dataset):
#     def __init__(self, data, seq_len):
#         super().__init__()
#         self.data = data
#         self.seq_len = seq_len

#     def __getitem__(self, index):
#         rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
#         full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
#         return full_seq

#     def __len__(self):
#         return self.data.size(0) // self.seq_len

# train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
# val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
# train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
# val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# # optimizer
# optim = AdamW(model.parameters(), lr=LEARNING_RATE)

# # training
# for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
#     model.train()

#     for __ in range(GRADIENT_ACCUMULATE_EVERY):
#         input_data = next(train_loader)
#         print("Shape of input data:", input_data.shape)
#         loss = model(input_data)
#         loss.mean().backward()

#     print(f"training loss: {loss.mean().item()}")
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#     optim.step()
#     optim.zero_grad()

#     if i % VALIDATE_EVERY == 0:
#         model.eval()
#         with torch.no_grad():
#             loss = model(next(val_loader))
#             print(f"validation loss: {loss.mean().item()}")

#     if i % GENERATE_EVERY == 0:
#         model.eval()
#         inp = random.choice(val_dataset)[:-1]
#         prime = decode_tokens(inp)
#         print("%s \n\n %s", (prime, "*" * 100))

#         sample = model.generate(inp[None, ...], GENERATE_LENGTH)
#         output_str = decode_tokens(sample[0])
#         print(output_str)
