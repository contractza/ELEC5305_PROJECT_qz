# ELEC5305_PROJECT_qz

for training the model, find one of the model for exmaple dprnn_enhanced.py, do:
python dprnn_enhanced.py 

the model will be saved and for evaluating the model, find the Corresponding evaluation file, for example, eval_enhanced.py, do:
python eval_enhanced.py
and then the result will saved in same file.














## Title: Voice separation and localization for multiple speakers

## Student information
Full name: Qihang Zhao <br>
Sid: 510022587 <br>
GitHub username: contractza <br>

## project overview
this project aims to research and implement multi-speaker speech separation and sound source localization technology. When two people are speaking simultaneously, how to sepearate each speaker's voice from a dualchannel recording and estimate their azimuth in space.
The reason for choosing this topic is that it integrates two fields: signal processing and deep learning. On one hand, it involves classic speech signal processing methods such as STFT and time-frequency masking; on the other hand, it utilizes the deep learning framework to improve the separation performance, possessing both engineering feasibility and academic exploration value.

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

## timetable:
week          	task
6–7	Literature research and download the MiniLibriMix dataset <br>
8–9	Build a MATLAB deep learning model and complete the STFT + LSTM mask prediction framework <br>
10–11	Model training and preliminary evaluation, and comparison of ICA and deep learning results <br>
12	Add the sound source location module to visualise <br>
13	final report and present video <br>

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







