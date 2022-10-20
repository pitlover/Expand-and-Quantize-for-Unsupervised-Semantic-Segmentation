import argparse
import os
import time
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from datasets.dataset import dataset_recon
from models.Discriminator import Discriminator_GL
from models.Generator import Generator
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from utils.loss import IDMRFLoss
from utils.utils import gaussian_weight


# Training
def train(args, gen, dis, opt_gen, opt_dis, epoch, train_loader):
    gen.train()
    dis.train()

    mse = nn.MSELoss(reduction='none').cuda(args.device)
    mrf = IDMRFLoss(device=args.device)

    for batch_idx, (I_l, I_r, I_m) in enumerate(train_loader):

        start_time = time.time()

        batchSize = I_l.shape[0]
        imgSize = I_l.shape[2]

        I_l, I_r, I_m = I_l.cuda(args.device), I_r.cuda(args.device), I_m.cuda(args.device)

        ## Generate Image
        I_pred, f_m, F_l, F_r, q_loss = gen(I_l, I_r)
        f_m_gt = gen(I_m, only_encode=True)  # gt for feature map of middle part
        I_pred_split = list(torch.split(I_pred, imgSize, dim=3))
        I_gt = torch.cat((I_l, I_m, I_r), 3)

        ## Discriminator
        fake = dis(I_pred)
        real = dis(I_gt)

        ## Compute losses
        # Pixel Reconstruction Loss
        weight = gaussian_weight(batchSize, imgSize, device=args.device)
        mask = weight + weight.flip(3)
        pixel_rec_loss = (mse(I_pred_split[0], I_l) + mse(I_pred_split[2], I_r) + mask * mse(I_pred_split[1],
                                                                                             I_m)).mean() * batchSize

        # Texture Consistency Loss (IDMRF Loss)
        mrf_loss = mrf((I_pred_split[1].cuda(args.device) + 1) / 2.0, (I_m.cuda(args.device) + 1) / 2.0) * 0.01

        # Feature Reconstruction Loss
        feat_rec_loss = mse(f_m, f_m_gt.detach()).mean() * batchSize

        # Feature Consistency Loss
        feat_cons_loss = (mse(F_l[0], F_r[0]) + mse(F_l[1], F_r[1]) + mse(F_l[2], F_r[2])).mean() * batchSize

        # RaLSGAN Adversarial Loss
        real_label = torch.ones(batchSize, 1).cuda(args.device)
        fake_label = torch.zeros(batchSize, 1).cuda(args.device)
        gen_adv_loss = ((fake - real.mean(0, keepdim=True) - fake_label) ** 2).mean() * batchSize * 0.002 * 0.9
        dis_adv_loss = (((real - fake.mean(0, keepdim=True) - real_label) ** 2).mean() + (
                    (fake - real.mean(0, keepdim=True) + real_label) ** 2).mean()) * batchSize

        gen_loss = pixel_rec_loss + mrf_loss.cuda(args.device) + feat_rec_loss + feat_cons_loss + q_loss + gen_adv_loss
        dis_loss = dis_adv_loss

        ## Update Generator
        if (batch_idx % 3) != 0:
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

        ## Update Discriminator
        if (batch_idx % 3) == 0:
            opt_dis.zero_grad()
            dis_loss.backward()
            opt_dis.step()

        print(
            f'Epoch:{epoch} | Batch:{batch_idx} | G_loss:{gen_loss:.2f} | D_loss:{dis_loss:.2f} | pixel_rec_loss:{pixel_rec_loss:.2f} | mrf_loss:{mrf_loss:.2f} | feat_rec_loss:{feat_rec_loss:.2f} | feat_cons_loss:{feat_cons_loss:.2f} | q_loss: {q_loss:.2f} | gen_adv_loss:{gen_adv_loss:.2f} |Time:{time.time() - start_time:.1f}s')

        if batch_idx % 10 == 0:
            with torch.no_grad():
                inputs = torch.cat((I_l[:4], torch.ones_like(I_l)[:4], I_r[:4]), dim=3)
                real_fake_images = torch.cat((inputs, I_pred[:4], I_gt[:4]))
                vutils.save_image(real_fake_images, os.path.join(args.save_weight_dir, f"Epoch{epoch}.jpg"), nrow=4)


if __name__ == '__main__':

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_batch_size', type=int, help='batch size of training data', default=20)
        parser.add_argument('--epochs', type=int, help='number of epoches', default=300)
        parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
        parser.add_argument('--alpha', type=float, help='learning rate decay for discriminator', default=0.1)
        parser.add_argument('--load_pretrain', type=int, help='whether to load pretrain weight', default=0)

        parser.add_argument('--skip_connection', type=int, help='layers with skip connection', nargs='+',
                            default=[0, 1, 2, 3, 4])
        parser.add_argument('--attention', type=int, help='layers with attention mechanism applied on skip connection',
                            nargs='+', default=[1])

        parser.add_argument('--load_weight_path', type=str, help='directory of pretrain model weights',
                            default='./checkpoints/model_latest.pt')
        parser.add_argument('--save_weight_dir', type=str, help='directory of saving model weights',
                            default='./checkpoints/')
        parser.add_argument('--train_data_dir', type=str, help='directory of training data')
        parser.add_argument('--device', type=int, help='which gpu', default=0)

        opts = parser.parse_args()
        return opts


    args = get_args()
    print(args)
    os.makedirs(args.save_weight_dir, exist_ok=True)

    # Initialize the model
    print('Initializing model...')
    gen = Generator(device=args.device, skip=args.skip_connection, attention=args.attention).cuda(args.device)
    dis = Discriminator_GL(imgSize=256).cuda(args.device)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=2e-5)
    opt_dis = optim.Adam(dis.parameters(), lr=args.lr * args.alpha, betas=(0.5, 0.9), weight_decay=2e-5)

    # Load pre-trained weight
    if args.load_pretrain:
        print('Loading model weight...')
        gen.load_state_dict(torch.load(args.load_weight_path)['gen'])
        dis.load_state_dict(torch.load(args.load_weight_path)['dis'])
        start_epoch = torch.load(args.load_weight_path)['epoch'] + 1
    else:
        start_epoch = 1

    # Load data
    print('Loading data...')
    transformations = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
    train_data = dataset_recon(root=args.train_data_dir, transforms=transformations, crop='rand', imgSize=256)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    print('train data: %d images' % (len(train_loader.dataset)))

    # Train & test the model
    for epoch in range(start_epoch, 1 + args.epochs):
        print("----Start training[%d]----" % epoch)
        train(args, gen, dis, opt_gen, opt_dis, epoch, train_loader)

        # Save the model weight
        save_dict = {'gen': gen.state_dict(), 'dis': dis.state_dict(), 'epoch': epoch}
        torch.save(save_dict, join(args.save_weight_dir, 'model_latest.pt'))
        if epoch % 10 == 0:
            torch.save(save_dict, join(args.save_weight_dir, f'model_epoch{epoch}.pt'))

