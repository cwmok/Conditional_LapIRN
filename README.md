# Conditional Deformable Image Registration with Convolutional Neural Network

This is the official Pytorch implementation of "Conditional Deformable Image Registration with Convolutional Neural Network" (MICCAI 2021), written by Tony C. W. Mok and Albert C. S. Chung.

## Prerequisites
- `Python 3.5.2+`
- `Pytorch 1.3.0 - 1.7.1`
- `NumPy`
- `NiBabel`

This code has been tested with `Pytorch 1.7.1` and NVIDIA TITAN RTX GPU.

## Inference
```
python Test_cLapIRN.py
```

The regularization weight can be changed by appending `--reg_input {{normalized weight within [0,1]}}` argument to the inference script. For example,
```
python Test_cLapIRN.py --reg_input 0.4
```
is equivalent to output the solution with regularization weight set to 4.

## Train your own model
Step 1: Replace `/PATH/TO/YOUR/DATA` with the path of your training data. You may also need to implement your own data generator (`Dataset_epoch` in `Functions.py`).

Step 2: Change the `imgshape` variable in `Train_cLapIRN.py` to match the resolution of your data.

Step 3: `python Train_cLapIRN.py` to train the model. Remember the data should be normalized within [0,1]. Otherwise, set `norm=True` in the provided data loader.

(Optional): Implement the custome validation code in line 368 at `Train_cLapIRN.py`. 

## Scalability/Out of memory error
1. You may adjust the size of the model by manipulating the argument `--start_channel` in `Train_cLapIRN.py` and `Test_cLapIRN.py`

2. You may adjust the number of conditional image registration module in `resblock_seq` function (at `Functions.py`). 


## Publication
If you find this repository useful, please cite:
- **Conditional Deformable Image Registration with Convolutional Neural Network**  
[Tony C. W. Mok](https://cwmok.github.io/ "Tony C. W. Mok"), Albert C. S. Chung  
MICCAI 2021. [arXiv (coming soon)]


## Acknowledgment
Some codes in this repository are modified from [IC-Net](https://github.com/zhangjun001/ICNet) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).


###### Keywords
Keywords: Conditional Image registration, Controllable Regularization, Deformable Image Registration
