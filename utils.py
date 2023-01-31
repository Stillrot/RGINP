import os
from os.path import join as ospj
import torch
import torch.nn as nn
import torchvision.utils as vutils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename):
    x = denormalize(x)
    if x.size()[1] == 4:
        x = x[:, 0:3, :, :]
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

@torch.no_grad()
def debug_image(models, args, sample_inputs, step):
    image = sample_inputs.image
    mask = sample_inputs.mask
    label = sample_inputs.label

    if mask.size()[1] == 4:
        mask = mask[:, 0:1, :, :]

    m_image = torch.mul(image, mask)
    g_mask = mask.repeat(1, 3, 1, 1)

    N = image.size(0)
    if N > 8:
        N = 7

    label = label - 0.5
    completion_image = models.LBAM_generator(m_image, g_mask, label, None, mode='enc_dec')

    if args.mode == 'test':
        filename1 = ospj(args.result_dir, '%06d_1_input.jpg' % (step))
    elif args.mode == 'val':
        filename_org = ospj(args.val_sample_dir, '%06d_0_origin.jpg' % (step))
        save_image(image, N+1, filename_org)
        filename1 = ospj(args.val_sample_dir, '%06d_1_input.jpg' % (step))
    # elif args.mode == 'train':
    else:
        raise NotImplementedError
    save_image(m_image, N+1, filename1)

    if args.mode == 'test':
        filename2 = ospj(args.result_dir, '%06d_2_completion.jpg' % (step))
    elif args.mode == 'val':
        filename2 = ospj(args.val_sample_dir, '%06d_2_completion.jpg' % (step))
    # elif args.mode == 'train':
    else:
        raise NotImplementedError
    save_image(completion_image, N+1, filename2)
    
        

