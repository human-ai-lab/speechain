# Overview

**A Machine Speech Chain Toolkit for ASR, TTS, and Both**

_SpeeChain_ is an open-source PyTorch-based speech and language processing toolkit produced by the [_AHC lab_](https://ahcweb01.naist.jp/en/) at Nara Institute of Science and Technology (NAIST). 
This toolkit is designed to simplify the pipeline of the research on the machine speech chain, 
i.e., the joint model of automatic speech recognition (ASR) and text-to-speech synthesis (TTS). 

_SpeeChain is currently in beta._ Contribution to this toolkit is warmly welcomed anywhere, anytime! 

If you find our toolkit helpful for your research, we sincerely hope that you can give us a star⭐! 
Anytime you encounter problems when using our toolkit, please don't hesitate to leave us an issue!

## Table of Contents
1. [**Machine Speech Chain**](https://github.com/bagustris/SpeeChain#machine-speech-chain)
2. [**Toolkit Characteristics**](https://github.com/bagustris/SpeeChain#toolkit-characteristics)
3. [**Directory Structure**](#directory-structure)
4. [**Quick Start**](#quick-start)
5. [**Work flow**](#work-flow)


## Machine Speech Chain
* Offline TTS→ASR Chain

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain#table-of-contents)


## Toolkit Characteristics
* **Data Processing:**
    - On-the-fly Log-Mel Spectrogram Extraction  
    - On-the-fly SpecAugment  
    - On-the-fly Feature Normalization  
* **Model Training:**
    - Multi-GPU Model Distribution based on _torch.nn.parallel.DistributedDataParallel_   
    - Real-time status reporting by online _Tensorboard_ and offline _Matplotlib_  
    - Real-time learning dynamics visualization (attention visualization, spectrogram visualization)  
* **Data Loading:**
    - On-the-fly mixture of multiple datasets in a single dataloader.  
    - On-the-fly data selection for each dataloader to filter the undesired data samples.  
    - Multi-dataloader batch generation is used to form training batches using multiple datasets.   
* **Optimization:**
    - Model training can be done by multiple optimizers. Each optimizer is responsible for a specific part -  model parameters.  
    - Gradient accumulation for mimicking the large-batch gradients by the ones on several small batches.  
    - Easy-to-set finetuning factor to scale down the learning rates without any modification of the scheduler configuration.  
* **Model Evaluation:**
    - Multi-level _.md_ evaluation reports (overall-level, group-level model, and sample-level) without any - yout misplacement.  
    - Histogram visualization for the distribution of evaluation metrics.  
    - Top N bad case analysis for better model diagnosis.

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain#table-of-contents)

## Directory Structure

```shell
├── config                # configuration for feature extraction
│   └── feat
├── CONTRIBUTING.md       # convention for contributor
├── create_env.sh         # bash shell to create environment
├── datasets              # dataset folder, put data here, make softlink, or set in config file
│   ├── data_dumping.sh
│   ├── librispeech
│   ├── libritts
│   ├── ljspeech
│   ├── meta_generator.py
│   ├── meta_post_processor.py
│   ├── mfa_preparation.sh
│   └── vctk
├── docs                  # folder to build docs
├── environment.yaml    
├── LICENSE              
├── recipes               # folder for experiment, you will work here
│   ├── asr
│   ├── lm
│   ├── offline_tts2asr
│   ├── run.sh
│   └── tts
├── requirements.txt
├── run.sh
├── scripts
│   └── gen_ref_pages.py
├── setup.py
└── speechain             # directory for speechain toolkit code
    ├── criterion
    ├── dataset
    ├── infer_func
    ├── iterator
    ├── model
    ├── module
    ├── monitor.py
    ├── optim_sche
    ├── pyscripts
    ├── runner.py
    ├── snapshooter.py
    ├── tokenizer
    └── utilbox
```

## Quick Start
Try minilibrispeech recipe `train-clean-5` in ASR.

# Workflow
We recommend you first install *Anaconda* into your machine before using our toolkit. 
After the installation of *Anaconda*, please follow the steps below to deploy our toolkit on your machine:

 1. Find a path with enough disk memory space. (e.g., at least 500GB if you want to use _LibriSpeech_ or _LibriTTS_ datasets).  
 2. Clone our toolkit by `git clone https://github.com/bagustris/SpeeChain.git`.  
 3. Go to the root path of our toolkit by `cd SpeeChain`.  
 4. Run `source envir_preparation.sh` to build the environment for SpeeChain toolkit.  
After execution, a virtual environment named `speechain` will be created and two environmental variables `SPEECHAIN_ROOT` and `SPEECHAIN_PYTHON` will be initialized in your `~/.bashrc`.  
**Note:** It must be executed in the root path `SpeeChain` and by the command `source` rather than `./envir_preparation.sh`.  
 5. Run `conda activate speechain` in your terminal to examine the installation of Conda environment. 
If the environment `speechain` is not successfully activated, please run `conda env create -f environment.yaml`, `conda activate speechain` and `pip install -e ./` to manually install it.  
 6. Run `echo ${SPEECHAIN_ROOT}` and `echo ${SPEECHAIN_PYTHON}` in your terminal to examine the environmental variables. If either one is empty, please manually add them into your `~/.bashrc` by `export SPEECHAIN_ROOT=xxx` or `export SPEECHAIN_PYTHON=xxx` and then activate them by `source ~/.bashrc`. 

    - `SPEECHAIN_ROOT` should be the absolute path of the `SpeeChain` folder you have just cloned (i.e. `/xxx/SpeeChain` where `/xxx/` is the parent directory); 
   
    - `SPEECHAIN_PYTHON` should be the absolute path of the python compiler in the folder of `speechain` environment (i.e. `/xxx/anaconda3/envs/speechain/bin/python3.X` where `/xxx/` is where your `anaconda3` is placed and `X` depends on `environment.yaml`).  

 7. Read the [handbook](./handbook) and start your journey in SpeeChain!  

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain#table-of-contents)
