import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from speech_recognition_module import SpeechRecognitionModel, scheduler, optimizer
from prepocess import training_data_processing, _blank_code
from comet_ml import Experiment
from train import train
# from evaluate import evaluate

def main(learning_rate=5e-4, epochs=10):

    hparams = {
        'n_cnn_layers': 3,
        'n_rnn_layers': 5,
        'rnn_dim': 512,
        'n_class': 29,
        'n_feats': 128,
        'stride': 2,
        'dropout': 0.1,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': 32
    }

    torch.manual_seed(7)
    device = torch.device('cpu')

    train_dataset = torchaudio.datasets.LIBRISPEECH('./train', url='train-clean-100')

    train_dataset = DataLoader(dataset=train_dataset,
                               batch_size=hparams['batch_size'],
                               shuffle=True,
                               collate_fn=lambda x: training_data_processing(x))

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'],
        hparams['dropout']).to(device)

    #  print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    experiment = Experiment(api_key='068a6rtlfjJeu9xgY5jdjAcgx',
                            project_name='fmi-ai',
                            workspace='ariolandi',)

    opt = optimizer(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=_blank_code).to(device)
    schd = scheduler(opt, hparams['learning_rate'],
                     int(len(train_dataset)),
                     hparams['epochs'],
                     'linear')

    for epoch in range(1, epochs + 1):
        train(model, device, train_dataset, criterion, opt, schd, epoch, experiment)


main()
