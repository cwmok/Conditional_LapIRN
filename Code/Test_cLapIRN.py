import os
from argparse import ArgumentParser

import numpy as np
import torch
from matplotlib import pyplot as plt
from ipywidgets import interact, IntSlider, fixed

import torch.nn.functional as F
import SimpleITK as sitk

from Functions import generate_grid_unit, save_img, save_flow, transform_unit_flow_to_flow, load_4D, imgnorm
from miccai2021_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit


parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/LDR_OASIS_NCC_unit_disp_add_fea7_reg01_10_testing_stagelvl3_60000.pth',
                    help="Trained model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Data/image_A.nii',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Data/image_B.nii',
                    help="moving image")
parser.add_argument("--reg_input", type=float,
                    dest="reg_input", default=0.4,
                    help="Normalized smoothness regularization (within [0,1])")
opt = parser.parse_args()

savepath = opt.savepath
fixed_path = opt.fixed
moving_path = opt.moving
if not os.path.isdir(savepath):
    os.mkdir(savepath)

start_channel = opt.start_channel
reg_input = opt.reg_input


def test():
    print("Current reg_input: ", str(reg_input))

    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()


    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
        
    fixed_img = load_4D(fixed_path)
    moving_img = load_4D(moving_path)
    
    #Added code to deal with reshape
    if np.shape(fixed_img) == np.shape(moving_img):
        ori_imgshape = np.shape(fixed_img[0])
    else:
        raise ValueError('Dimensions of fixed_img and moving_img must be the same')
    
    grid_full = generate_grid_unit(ori_imgshape)
    grid_full = torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape)).cuda().float()
    
    # TODO
    # Check that axes are coherent with imgshape dims and roll axes if not. 
    ###
    
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # normalize image to [0, 1]
    norm = True
    if norm:
        fixed_img = imgnorm(fixed_img)
        moving_img = imgnorm(moving_img)

    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    with torch.no_grad():

        reg_code = torch.tensor([reg_input], dtype=fixed_img.dtype, device=fixed_img.device).unsqueeze(dim=0)
        
        moving_img_down = F.interpolate(moving_img, size=imgshape, mode='trilinear')
        fixed_img_down = F.interpolate(fixed_img, size=imgshape, mode='trilinear')
        
        F_X_Y = model(moving_img_down, fixed_img_down, reg_code)
        F_X_Y = F.interpolate(F_X_Y, size=ori_imgshape, mode='trilinear', align_corners=True)

        X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid_full).data.cpu().numpy()[0, 0, :, :, :]

        F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

        save_flow(F_X_Y_cpu, savepath + '/warpped_flow_' + 'reg' + str(reg_input) + '.nii.gz')
        save_img(X_Y, savepath + '/warpped_moving_' + 'reg' + str(reg_input) + '.nii.gz')

    print("Result saved to :", savepath)


if __name__ == '__main__':
    imgshape = (160, 192, 144)
    imgshape_4 = (160 / 4, 192 / 4, 144 / 4)
    imgshape_2 = (160 / 2, 192 / 2, 144 / 2)

    range_flow = 0.4
    test()
