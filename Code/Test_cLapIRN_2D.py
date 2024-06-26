import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from Functions import save_img, save_flow, load_4D_with_header, imgnorm
from miccai2021_model_2D import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit


parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/LDR_OASIS_NCC_unit_disp_add_fea64_reg01_10_2D_stagelvl3_90000.pth',
                    help="Trained model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=64,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Data/image_A_2D.nii.gz',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Data/image_B_2D.nii.gz',
                    help="moving image")
parser.add_argument("--reg_input", type=float,
                    dest="reg_input", default=0.1,
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

    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 2, start_channel, is_train=True, imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 2, start_channel, is_train=True, imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 2, start_channel, is_train=False, imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()


    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    grid = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + imgshape, align_corners=True).cuda()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    fixed_img, header, affine = load_4D_with_header(fixed_path)
    moving_img, _, _ = load_4D_with_header(moving_path)

    fixed_img, moving_img = imgnorm(fixed_img), imgnorm(moving_img)
    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0).squeeze(-1)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0).squeeze(-1)

    with torch.no_grad():
        reg_code = torch.tensor([reg_input], dtype=fixed_img.dtype, device=fixed_img.device).unsqueeze(dim=0)

        F_X_Y = model(moving_img, fixed_img, reg_code)

        X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 1), grid).unsqueeze(-1).data.cpu().numpy()[0, 0, :, :, :]

        F_X_Y_cpu = F_X_Y.unsqueeze(-1).data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        x, y, z, _ = F_X_Y_cpu.shape
        F_X_Y_cpu[:, :, :, 0] = F_X_Y_cpu[:, :, :, 0] * (y - 1) / 2
        F_X_Y_cpu[:, :, :, 1] = F_X_Y_cpu[:, :, :, 1] * (x - 1) / 2

        save_flow(F_X_Y_cpu, savepath + '/warpped_flow_2D_' + 'reg' + str(reg_input) + '.nii.gz', header=header, affine=affine)
        save_img(X_Y, savepath + '/warpped_moving_2D_' + 'reg' + str(reg_input) + '.nii.gz', header=header, affine=affine)

    print("Result saved to :", savepath)


if __name__ == '__main__':
    imgshape = (160, 192)
    imgshape_4 = (160 // 4, 192 // 4)
    imgshape_2 = (160 // 2, 192 // 2)

    range_flow = 0.4
    test()
