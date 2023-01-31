import os
from os.path import join as ospj
import random
from PIL import Image
import numpy as np
from munch import Munch
import time
import datetime
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import completion_model
import dataloader as dl
from checkpoint import CheckPoint
import utils
from loss import compute_D_loss, compute_G_loss

class RGINP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.models = completion_model.build_model(args)

        self.models.LBAM_generator.to(self.device)
        self.models.discriminator.to(self.device)
        self.models.classifier.to(self.device)
        self.models.label_predict.to(self.device)

        self.optims = Munch()
        for model in self.models.keys():
            self.optims[model] = torch.optim.Adam(params = self.models[model].parameters(), 
                lr=args.d_lr if model in ['discriminator', 'classifier'] else args.g_lr,
                betas=[args.beta1, args.beta2] if model in ['discriminator', 'classifier'] else [0.5, 0.9])

        self.ckptios = [
            CheckPoint(ospj(args.checkpoint_dir, '{0:0>6}_models.ckpt'), **self.models),
            CheckPoint(ospj(args.checkpoint_dir, '{0:0>6}_optims.ckpt'), **self.optims)
            ]

        for name, model in self.models.items():
            print('Initializing %s...' % name)
            model.apply(utils.he_init)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)            
        
    def train(self, args):
        optims = self.optims

        print('train dataload')
        tr_loader = dl.dataset_loader(args, 'train')
        train_loader = DataLoader(dataset=tr_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)
        fetcher = dl.InputFetcher(train_loader)

        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        self.models.LBAM_generator.train()
        self.models.discriminator.train()
        self.models.classifier.train()
        self.models.label_predict.train()

        start_time = time.time()
        for epoch in range(args.resume_iter, args.total_iters):
            inputs = next(fetcher)
            image = inputs.image
            label = inputs.label
            mask = inputs.mask

            m_image = torch.mul(image, mask)

            # D train
            d_loss, d_loss_group = compute_D_loss(self.models, self.args, image, m_image, mask, label, self.device)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()
            optims.classifier.step()

            # G train
            g_loss, g_loss_group = compute_G_loss(self.models, self.args, image, m_image, mask, label, self.device)
            self._reset_grad()
            g_loss.backward()
            optims.LBAM_generator.step()
            optims.label_predict.step()
        
            if (epoch + 1) % args.verbose_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, epoch + 1, args.total_iters)

                all_losses = dict()
                for loss, prefix in zip([d_loss_group,  g_loss_group], ['  D/_', '  G/_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            if (epoch + 1) % args.save_step == 0:
                self._save_checkpoint(step=epoch + 1)

    @torch.no_grad()
    def test(self, args):
        test_models = self.models
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        # gt label
        att_list = open(self.args.label_dir, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in args.attrs]
        labels_attr = np.loadtxt(self.args.label_dir, skiprows=2, usecols=atts, dtype=np.int)

        images_name = os.listdir(args.image_test_src_dir)

        for image_name in images_name:
            # The file name of the mask and reference image must be the same as the source image.
            src_img = Image.open(os.path.join(args.image_test_src_dir, image_name))
            ref_img = Image.open(os.path.join(args.image_test_ref_dir, image_name))
            mask = Image.open(os.path.join(args.masks_test_dir, image_name))

            img_num = int(image_name.replace('.jpg', ''))
            original_attrs = labels_attr[img_num]
            original_attrs = torch.tensor((original_attrs + 1) // 2)
            original_attrs = original_attrs.view(1, original_attrs.size(0)).to(self.device)

            img_transform = transforms.Compose([transforms.Resize(size=args.img_size), transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            src_img = img_transform(src_img)
            ref_img = img_transform(ref_img).to(self.device)

            mask_transform = transforms.Compose([transforms.Resize(size=args.img_size), transforms.ToTensor()])
            mask = mask_transform(mask)

            if mask.size()[1] == 4:
                mask = mask[:, 0:1, :, :]

            m_image = torch.mul(src_img, mask)
            m_image = m_image.view(1, m_image.size(0), m_image.size(1), m_image.size(2)).to(self.device)
            g_mask = mask.view(1, mask.size(0), mask.size(1), mask.size(2)).to(self.device)

            N = src_img.size(0)
            if N > 8:
                N = 7

            ori_comp_image = test_models.LBAM_generator(m_image, g_mask, original_attrs, None, mode='enc_dec')
            pred_lbl = test_models.label_predict(ref_img.unsqueeze(0))
            ref_comp_image = test_models.LBAM_generator(m_image, g_mask, pred_lbl, None, mode='enc_dec')

            filename = ospj(args.result_dir, '%06d_01input_'% (img_num) + image_name)
            utils.save_image(m_image, N+1, filename)

            filename2 = ospj(args.result_dir, '%06d_02gt_'% (img_num) + image_name)
            utils.save_image(src_img, N+1, filename2)

            filename3 = ospj(args.result_dir, '%06d_03ori_label_'% (img_num) + image_name)
            utils.save_image(ori_comp_image, N+1, filename3)

            filename4 = ospj(args.result_dir, '%06d_04ref_label_'% (img_num) + image_name)
            utils.save_image(ref_comp_image, N+1, filename4)

    @torch.no_grad()
    def val(self, args):
        # only validate inpainting quality
        args = self.args
        val_models = self.models
        os.makedirs(args.val_sample_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)
        vl_loader = dl.dataset_loader(args, 'val')
        val_loader = DataLoader(dataset=vl_loader, batch_size=args.batch_size, num_workers=4, shuffle=False)
        fetcher = dl.InputFetcher(val_loader)

        tmp = random.randint(0, 2000 // args.batch_size)
        for _ in range(tmp):
            _ = next(fetcher)
        inputs = next(fetcher)

        print('Working on {}...'.format(ospj(args.val_sample_dir, 'validation.jpg')))
        utils.debug_image(val_models, args, inputs, args.resume_iter)