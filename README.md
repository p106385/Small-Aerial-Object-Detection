# FPGAN+ESN Model
## Model Architecture
<img src="Detection_Results/P-01.jpg" width="90%"></img>

## Dependencies and Installation
- Python 3 (Recommend to use Anaconda)
- PyTorch >= 1.0
- NVIDIA GPU + CUDA
- Python packages: `pip install -r path/to/requirement.txt`
## Training
`python train.py -c config_GAN.json`
## Testing
`python test.py -c config_GAN.json`
## Dataset Download links
1. **[DOTA Dataset](https://captain-whu.github.io/DOTA/index.html)** - DOTA dataset from here.

2. **[OGST Dataset](https://data.mendeley.com/datasets/bkxj8z84m9/3)** - OGST dataset from here.

3. **[COWC Dataset](https://gdo152.llnl.gov/cowc/download/cowc-m/datasets/)** - COWC dataset from here.

## Generate High, Low, and Bicubic Resolution Images For  Dataset
Download pre-made dataset of COWC from [here](https://gdo152.llnl.gov/cowc/download/cowc-m/datasets/DetectionPatches_256x256.tgz) and this file (scripts_GAN_HR-LR.py)  can be used with pre-made dataset to create high/low-resolution and bicubic images. Make sure to copy annotation files (.txt) in the HR, LR and Bic folder.  


## Configure JSON File

To get started, update the directory paths in the JSON file to match your local user environment. Pretrained weights can be downloaded from [Google Drive here](https://drive.google.com/drive/folders/15xN_TKKTUpQ5EVdZWJ2aZUa4Y-u-Mt0f?usp=sharing).


```json
{
    "data_loader": {
        "type": "COWCGANFrcnnDataLoader",
        "args": {
            "data_dir_GT": "/data2/PhD/test/GAN/DOTA_images/HR_2/",
            "data_dir_LQ": "/data2/PhD/test/GAN/DOTA_images/LR_2/",
            "batch_size": 1,
            "shuffle": true,
            "validation_split": 0.0,
            "num_workers": 2
        }
    },
    "path": {
        "models": "/data2/PhD/test/GAN/saved/pretrained_models_FPGAN_ESN",
        "pretrain_model_G": "/data2/PhD/test/GAN/saved/pretrained_models_FPGAN_ESN/20000_G1.pth",
        "pretrain_model_D": "/data2/PhD/test/GAN/saved/pretrained_models_FPGAN_ESN/20000_D1.pth",
        "pretrain_model_FRCNN": "/data2/PhD/test/GAN/saved/pretrained_models_FPGAN_ESN/20000_FRCNN1.pth",
        "data_dir_Valid": "//data2/PhD/test/GAN/DOTA_images/LR_2/",
        "Test_Result_SR": "/data2/PhD/test/GAN/Results",
        "strict_load": false,
        "log": "saved/logs"
    },
    "logger": {
        "print_freq": 1,
        "save_checkpoint_freq": 20000
    }
}
