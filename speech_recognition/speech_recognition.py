from comet_ml import Experiment
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from speech_recognition.speech_recognition_module import SpeechRecognitionModel, scheduler, optimizer
from speech_recognition.prepocess import training_data_processing, _blank_code
from speech_recognition.train import train
# from evaluate import evaluate


def speech_recognition(srt_directory, wav_directory, learning_rate=5e-4, epochs=10):

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

    train_dataset = torchaudio.datasets.LIBRISPEECH('./train',
                                                    url='train-clean-100',
                                                    download=True)

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

    comet_api_key = "068a6rtlfjJeu9xgY5jdjAcgx"  # add your api key here
    project_name = "fmi-ai"
    experiment_name = "speechrecognition-colab"
    experiment = ""
    if comet_api_key:
        experiment = Experiment(api_key=comet_api_key, project_name=project_name, parse_args=False)
        experiment.set_name(experiment_name)
        experiment.display()
    else:
        experiment = Experiment(api_key='dummy_key', disabled=True)

    opt = optimizer(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=_blank_code).to(device)
    schd = scheduler(opt, hparams['learning_rate'],
                     int(len(train_dataset)),
                     hparams['epochs'],
                     'linear')

    for epoch in range(1, epochs + 1):
        train(model, device, train_dataset, criterion, opt, schd, epoch, experiment)

    # evaluate(srt_directory, wav_directory)


speech_recognition('', '')
