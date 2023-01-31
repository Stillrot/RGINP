import argparse
from torch.backends import cudnn
import torch
from rginp import RGINP

IMAGE_SIZE = 256

def main(args):
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    rginp_ = RGINP(args)

    if args.mode == 'train':
        rginp_.train(args)

    elif args.mode == 'val':
        rginp_.val(args)

    elif args.mode == 'test':
        rginp_.test(args)

    else:
        raise NotImplementedError

attrs_default = ['Bushy_Eyebrows', 'Mouth_Slightly_Open', 'Big_Lips', 'Male', 'Mustache', 'Young', 'Smiling', 'Wearing_Lipstick', 'No_Beard']
attrs_len = len(attrs_default)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=7777, help='random seed')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val', 'test'], help='This argument is used in solver')

    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size')

    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--n_attrs', default=attrs_len, type=int)
    parser.add_argument('--total_iters', type=int, default=200000, help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training/testing')
    parser.add_argument('--d_lr', type=float, default=0.00001)
    parser.add_argument('--g_lr', type=float, default=0.0001)

    parser.add_argument('--lambda_valid', type=float, default=1, help='loss for valid area')
    parser.add_argument('--lambda_hole', type=float, default=6, help='loss for hole area')
    parser.add_argument('--lambda_ssim', type=float, default=3, help='MS SSIM loss')
    parser.add_argument('--lambda_prc', type=float, default=0.01, help='perceptual loss')
    parser.add_argument('--lambda_style', type=float, default=120, help='vgg style loss')
    parser.add_argument('--lambda_recon', type=float, default=0.5, help='recon loss')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='adv loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='gp loss')

    parser.add_argument('--beta1', type=float, default=0.0, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='Decay rate for 2nd moment of Adam')

    parser.add_argument('--image_dir', type=str, default='../data/train/CelebA_HQ', help='image directory')
    parser.add_argument('--label_dir', type=str, default='../data/train/CelebAMask-HQ-attribute-anno.txt', help='label directory')
    parser.add_argument('--masks_dir', type=str, default='../data/masks/train', help='masks directory')
    
    parser.add_argument('--image_val_dir', type=str, default='../data/val/CelebA_HQ', help='validation image directory')
    parser.add_argument('--masks_val_dir', type=str, default='../data/masks/test', help='validation masks directory')

    parser.add_argument('--image_test_src_dir', type=str, default='user_test/user_input/image/src', help='test src image directory')
    parser.add_argument('--image_test_ref_dir', type=str, default='user_test/user_input/image/ref', help='test src image directory')
    parser.add_argument('--masks_test_dir', type=str, default='user_test/user_input/mask', help='test mask directory')

    parser.add_argument('--checkpoint_dir', type=str, default='ckpt', help='model checkpoint directory')
    parser.add_argument('--val_sample_dir', type=str, default='val_sample', help='validation sample directory')
    parser.add_argument('--result_dir', type=str, default='user_test/test_result', help='test sample save directory')

    parser.add_argument('--save_step', type=int, default=5000)
    parser.add_argument('--verbose_step', type=int, default=50)

    args = parser.parse_args()
    args.img_size = IMAGE_SIZE
    main(args)