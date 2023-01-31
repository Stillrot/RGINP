from munch import Munch
import torch.nn as nn
from torchvision import models
import torch
from torch import autograd
import pytorch_msssim


def compute_G_loss(models, args, image, m_image, mask, label, device):
    label = label - 0.5
    g_mask = mask.repeat(1, 3, 1, 1)

    enc_feature = models.LBAM_generator(m_image, g_mask, None, None, mode='enc')
    completion_image = models.LBAM_generator(m_image, g_mask, label, enc_feature, mode='dec')

    ori_pred_lbl = models.label_predict(image.to(device))
    pred_lbl_completion_image = models.LBAM_generator(m_image, g_mask, ori_pred_lbl, enc_feature, mode='dec')

    with torch.no_grad():
        cmp_D = models.discriminator(pred_lbl_completion_image)
        cls_D = models.classifier(pred_lbl_completion_image)

    # WGAN_GP: fake to real
    adv_fake_loss = cmp_D.mean().sum() * 1

    # completion loss
    extractor = VGG16FeatureExtractor().to(device)

    l1 = nn.L1Loss()
    # L1 loss
    hole_loss = l1((1 - mask) * completion_image, (1 - mask) * image)
    valid_loss = l1(mask * completion_image, mask * image)
    output_comp = mask * image + (1 - mask) * completion_image

    feat_output_comp = extractor(output_comp)
    feat_output = extractor(completion_image)
    feat_gt = extractor(image)

    # vgg perceptual loss
    prc_loss = 0.0
    for i in range(3):
        prc_loss += l1(feat_output[i], feat_gt[i])
        prc_loss += l1(feat_output_comp[i], feat_gt[i])

    # vgg style loss
    vgg_style_loss = 0.0
    for i in range(3):
        vgg_style_loss += l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
        vgg_style_loss += l1(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i]))

    # MS-SSIM LOSS
    loss_ms_ssim = pytorch_msssim.MS_SSIM(data_range=1)
    loss_ms_ssim.to(device)
    loss_ms_ssim_value = (1 - loss_ms_ssim((output_comp + 1) * 0.5, (image + 1) * 0.5))

    MSELoss = torch.nn.MSELoss()
    label_loss = MSELoss(cls_D, ori_pred_lbl.float())

    loss = args.lambda_adv * adv_fake_loss \
           + args.lambda_hole * 0.5 * hole_loss \
           + args.lambda_valid * 0.5 * valid_loss \
           + args.lambda_ssim * loss_ms_ssim_value \
           + args.lambda_prc * prc_loss \
           + args.lambda_style * vgg_style_loss \
           + label_loss

    return loss, Munch(fake=adv_fake_loss, SSIM=loss_ms_ssim_value,
                       hole=hole_loss, valid=valid_loss,
                       precep=prc_loss, vgg_sty=vgg_style_loss, pred_lbl=label_loss)


def compute_D_loss(models, args, image, m_image, mask, label, device):
    label = label - 0.5
    g_mask = mask.repeat(1, 3, 1, 1)

    with torch.no_grad():
        enc_feature = models.LBAM_generator(m_image, g_mask, None, None, mode='enc')
        ori_pred_lbl = models.label_predict(image.to(device))
        completion_image = models.LBAM_generator(m_image, g_mask, ori_pred_lbl, enc_feature, mode='dec')

    fake_rand_D = models.discriminator(completion_image)

    image.requires_grad_()
    ori_D = models.discriminator(image)
    real_cls = models.classifier(image)

    # WGAN-GP real to real
    adv_real_loss = ori_D.mean().sum() * -1

    # WGAN-GP fake to fake
    adv_fake_loss = fake_rand_D.mean().sum() * 1

    loss_gp = calc_gradient_penalty(models.discriminator, image, completion_image, mask, torch.cuda.is_available(),
                                    args.lambda_gp)

    MSELoss = torch.nn.MSELoss()
    lbl_loss = MSELoss(real_cls, label.float())

    loss = adv_fake_loss - adv_real_loss + loss_gp + lbl_loss
    return loss, Munch(real=adv_real_loss, fake=adv_fake_loss, gp=loss_gp, lbl_D=lbl_loss)


def calc_gradient_penalty(netD, real_data, fake_data, masks, cuda, Lambda):
    """
    https://github.com/jalola/improved-wgan-pytorch
    """
    BATCH_SIZE = real_data.size()[0]
    DIM = real_data.size()[2]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    if cuda:
        alpha = alpha.cuda()

    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    if cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty.sum().mean()


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]