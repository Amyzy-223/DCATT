# DCATT
## Introduction
The code for the DCATT (diffusion convolutional attention model), a novel spatial-temporal model to provide multistep predictions of operators' abnormal cognitive state under heat stress. DCATT effectively utilizes spatiotemporal information in fNIRS dynamic graphs and ECG time-frequency features extracted via neuroscience analysis. The model captures spatial dependencies via diffusion graph convolution to simulate dynamic brain networks, enhancing short-term forecasting. It further models temporal relationships by a multi-head self-attention mechanism to aggregate historical information, improving long-term prediction. Eventually, a multi-head cross-attention mechanism with position encoding is applied to physiological embeddings to generate latent future cognitive representations.

## Content structure
DCATT was trained and evaluated on two datasets and tasks.

### main_cog: 

#### dataset instruction
A dataset derived from the experiment published in "Time course of cognitive functions and physiological response under heat exposure: Thermal tolerance exposure time based on ECG and fNIRS". Thirty-eight volunteers participated in the experiments involving seven hot-humid exposure scenarios (25℃ to 44℃) to explore the impact of heat pressure and exposure time on operators' cognitive functions. The accuracy and reaction time of classical cognitive tests, fNIRS signals, and ECG signals were recorded simultaneously during the experiment. The input

#### how to train it
run './main_cog/main.py'

#### prepare work
For first use, please download the dataset file from Google Drive and place it in './data/dataset/'.
Google Drive URL: https://drive.google.com/file/d/1n-8RLCe7YILVuSeDmbj-gVPhkMGkHtkU/view?usp=drive_link


### main_seed:
#### dataset instruction
SEED dataset, a classic dataset in the emotion recognition domain.

#### how to train it
run 'main_seed/main.py'

#### prepare work:
