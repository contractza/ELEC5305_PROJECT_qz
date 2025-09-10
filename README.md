# ELEC5305_PROJECT_qz

## Title: Voice separation and localization for multiple speakers

## Student information
Full name: Qihang Zhao
Sid: 510022587
GitHub username: contractza


this project aims to research and implement multi-speaker speech separation and sound source localization technology. When two people are speaking simultaneously, how to sepearate each speaker's voice from a dualchannel recording and estimate their azimuth in space.

## research problems:
- How can deep learning methods be used to separate the signals of individual speakers from mixed speech?
- Compared with traditional methods (ICA), are deep learning models better in separation effects?
- Can dual-channel recording further assist in sound source localization and thereby enhance the practicality of the system?

## method
The STFT features of mixed speech are modeled by using the LSTM/BiLSTM network.
model input: magnitude spectrogram of mixed speech
output: time-frequency masks for each speaker
multiply the predicted mask with the mixed speech and then reconstruct the separated speech through ISTFT.
loss function: SDR/MSE

## except result
Realize the separation of voices between two speakers, the result is better than ICA baseline


## dataset:
MiniLibriMix (https://zenodo.org/records/3871592), includes dual speaker mixed voice
use dual-channel microphones to record the dialogue between two people for demonstration and testing the generalization ability of the model



## Project Submission Details
The project submission consists of three parts:
(1) Working code that can be downloaded via github
(2) A written project report.
(3) A brief video demonstrating your code in action.


## reference


[1] DeLiang Wang, Jitong Chen, 2017, Supervised Speech Separation Based on Deep Learning: An Overview, https://arxiv.org/abs/1708.07524 <br>
[2] MatLab: Cocktail Party Source Separation Using Deep Learning Networks, https://www.mathworks.com/help/audio/ug/cocktail-party-source-separation-using-deep-learning-networks.html <br>
[3] Yi Luo, Nima Mesgarani, 2018, Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation, https://arxiv.org/abs/1809.07454 <br>
[4] https://blog.csdn.net/m0_56942491/article/details/134455964 <br>







