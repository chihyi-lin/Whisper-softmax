Guarding Against Deepfake Audio with One-Class Softmax
===============
The code in this repository is adapted from [deepfake-whisper-features](https://github.com/piotrkawa/deepfake-whisper-features).

This repository is mainly used for the experiment of MFCC+Whisper+MesoNet-Softmax from the project [Guarding Against Deepfake Audio with One-Class Softmax](https://github.com/chihyi-lin/One-Class-Softmax). 

## Requirements
(Follow the original deepfake-whisper-features repo.)
### Whisper
To download Whisper encoder used in training run `download_whisper.py`.

### Dependencies
Install required dependencies using:
```bash
bash install.sh
```

List of requirements:
```
python=3.8
pytorch==1.11.0
torchaudio==0.11
asteroid-filterbanks==0.4.0
librosa==0.9.2
openai whisper (git+https://github.com/openai/whisper.git@7858aa9c08d98f75575035ecd6481f462d66ca27)
```


### Configs

Both training and evaluation scripts are configured with the use of `.yaml` configuration files.
e.g.:
```yaml
data:
  seed: 42

checkpoint:
  path: ""

model:
  name: "whisper_frontend_mesonet"
  parameters:
    freeze_encoder: True
    input_channels: 2
    fc1_dim: 1024
    frontend_algorithm: ["lfcc"]
  optimizer:
    lr: 0.0001
    weight_decay: 0.0001
```

Other example configs are available under `configs/training/`.

## Train and test pipeline 

To perform full pipeline of training and testing please use `train_and_test.py` script.

```
usage: train_and_test.py [-h] [--asv_path ASV_PATH] [--in_the_wild_path IN_THE_WILD_PATH] [--config CONFIG] [--train_amount TRAIN_AMOUNT] [--test_amount TEST_AMOUNT] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--ckpt CKPT] [--cpu]

Arguments: 
    --asv_path          Path to the ASVSpoof2021 DF root dir
    --in_the_wild_path  Path to the In-The-Wild root dir
    --config            Path to the config file
    --train_amount      Number of samples to train on (default: 100000)
    --valid_amount      Number of samples to validate on (default: 25000)
    --test_amount       Number of samples to test on (default: None - all)
    --batch_size        Batch size (default: 8)
    --epochs            Number of epochs (default: 10)
    --ckpt              Path to saved models (default: 'trained_models')
    --cpu               Force using CPU
    --add_loss          ocsoftmax
```

E.g., In my experiment, MFCC+Whisper(frozen)+MesoNet is first trained for 5 epochs:
```bash
CUDA_VISIBLE_DEVICES=[idx] python train_and_test.py --asv_path [file_path] --in_the_wild_path [file_path] --config configs/training/whisper_frontend_mesonet_mfcc.yaml --batch_size 8 --epochs 5 --train_amount 100000 --valid_amount 25000
```


## Finetune and test pipeline
E.g., Continuously finetune the trained MFCC+Whisper(frozen)+MesoNet model for 15 epochs, with Whisper unfrozen: 
```
CUDA_VISIBLE_DEVICES=[idx] python train_and_test.py --asv_path [file_path] --in_the_wild_path [file_path] --config [configs/trained_model.yaml] --batch_size 8 --epochs 15 --train_amount 100000 --valid_amount 25000
```

## Only evaluation
E.g., Evaluate the trained model:
```
python evaluate_models.py --in_the_wild_path [file_path] --config configs/model__whisper_frontend_mesonet__1712814576.3199446.yaml
```
