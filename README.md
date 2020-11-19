# Deep-learning-for-SEC-10-Q-itemizing

## Environment configuration

1. Ubuntu 16.04 or upper (need a nvidia GPU).
2. Pytorch with a CUDA. https://dev.to/evanilukhin/guide-to-install-pytorch-with-cuda-on-ubuntu-18-04-5217 .

## Python package setup
1. pytorch                            1.6.0
2. torchvision                        0.7.0
3. torchnet                           0.0.4
4. visdom                             0.1.8.9
5. scipy                              1.5.0
6. ipdb                               0.13.3

## Train data location
1. Unzip "more_good_img.zip" and "more_bad_img.zip" to "./data/text_data/" in this repository. 

## Train model
1. Run the commend "python -m visdom.server -port 8099" to start a visdom server to visualize.
2. Run "python train.py" to train the model.
3. Model checkpoints will be saved in the "./checkpoints/"

## Test model download
1. Download trained model from "https://drive.google.com/drive/folders/1DSQTTkYGYVvKLzT_D3oQVW4VHFq7BBlk?usp=sharing" and put it in the "./checkpoints/"

## Test model
1. Run the commend "python -m visdom.server -port 8099" to start a visdom server to visualize. 
2. Put test data (*.png files) to "./data/test_text_data/".
3. Run "python test.py" to generate a result file, where each line contains test file's name and predictive label.

