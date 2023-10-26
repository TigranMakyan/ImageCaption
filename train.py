import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from utils import load_checkpoint, save_chackpoint, print_examples
from model import CNNtoRNN
from data import get_loader

def train():
    transform = T.Compose([
        T.Resize((356, 356)),
        T.RandomCrop((299, 299)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader, dataset = get_loader(
        root_folder='/home/user/Datasets/flickr8k/images',
        annotation_file='/home/user/Datasets/flickr8k/captions.txt',
        transform=transform,
        num_workers=8,
    )
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False
    save_model = True

    # hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)

    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 50

    # for tensorboard
    writer = SummaryWriter('runs/flickr')
    step = 0

    # initialize model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load('my_checkpoint.pth'), model, optimizer)
    
    model.train()
    print(f'Loader len is {len(train_loader)}')
    for epoch in range(num_epochs):
        # print_examples(model, device, dataset)
        print(f'Epoch {epoch}')
        if save_model:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
            }
            save_chackpoint(checkpoint)

        for idx, (imgs, captions) in enumerate(train_loader):
            print(f'index of loader: {idx}')
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            # (seq_len, N, vocab_size),  (seq_len, N)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            writer.add_scalar('Training loss', loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    train() 
