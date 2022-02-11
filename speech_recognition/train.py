import torch
from speech_recognition.prepocess import DataManipulation
from torch.nn import functional as F


def train(model, device, train_loader, criterion,
          optimizer, scheduler, epoch, experiment):
    model.train()
    data_len = len(train_loader)
    with experiment.train():
        for batch_idx, data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = DataManipulation.transpose_position(output, 0)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            experiment.log_metric('loss', loss.item())
            experiment.log_metric('learning_rate', scheduler.get_lr())

            optimizer.step()
            scheduler.step()

            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))

    torch.save(model.state_dict(), './')
