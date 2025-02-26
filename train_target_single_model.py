# Developed by Yuekai Xu, Aaron Honjaya, Zixuan Liu, all rights reserved to GrInAdapt team.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='1')
parser.add_argument('--model-file', type=str, default='./models/oneNorm/278.pth')
parser.add_argument('--file_name', type=str, default='Evaluation_image_level_model')
parser.add_argument('--fail_image_path', type=str, default='./fail_image_list.csv')
parser.add_argument('--save_root', type=str, default='./log_results/')
parser.add_argument('--model', type=str, default='IPN_V2', help='IPN_V2')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr-decrease-rate', type=float, default=0.9, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')

parser.add_argument('--data-dir', default='')
parser.add_argument('--dataset', type=str, default='AIREADI')
parser.add_argument('--model-source', type=str, default='OCTA500')
parser.add_argument('--batch-size', type=int, default=2)

#test155dsad
parser.add_argument('--model-ema-rate', type=float, default=0.995)
parser.add_argument('--pseudo-label-threshold', type=float, default=0.5)
parser.add_argument('--mean-loss-calc-bound-ratio', type=float, default=0.2)

# OCTA 500 args
parser.add_argument("--in_channels", type=int, default=2, help="input channels")
parser.add_argument("--n_classes", type=int, default=5, help="class number")
parser.add_argument("--method", type=str, default="IPN_V2", help="IPN, IPN_V2")
parser.add_argument("--ava_classes", type=int, default=2, help="label channels")
parser.add_argument("--proj_map_channels", type=int, default=2, help="class number")
parser.add_argument("--get_2D_pred", type=bool, default=True, help="get 2D head")
parser.add_argument("--proj_train_ratio", type=int, default=1, help="proj_map H or W to train_size H or W ratio. Currently only supports 1 or 2")
parser.add_argument("--dc_norms", type = str, default = "NG", help="normalization for Double Conv")

parser.add_argument('--checkpoint-interval', type=int, default=300,
                    help='Save model checkpoint every K patient updates')
parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Path to resume checkpoint')
parser.add_argument("--run_all_success", type=bool, default=False, help="run the training and testing for all the success cases")
parser.add_argument("--mask_optic_disc", type=bool, default=False, help="mask out the optic disc")
parser.add_argument('--annealing_factor', type=str, default=None, help='annealing factor for the loss')

args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import torch.backends.cudnn as cudnn
import random
import sys
import json
from tqdm import tqdm


# OCTA 500 import
import model as s_model
from dataloaders.aireadi_dataloader import AireadiSegmentation, ResumeSampler
from dataloaders.custom_octa_transform import Custom3DTransformTrain, Custom3DTransformWeak
from training_utils import DiceLoss
from eval_image import eval_final, summarize_csv_metrics

# test 11212
seed = 42
savefig = False
get_hd = True
model_save = True
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def colorize_segmentation(seg_idx):
    if isinstance(seg_idx, torch.Tensor):
        seg_idx = seg_idx.cpu().numpy()

    H, W = seg_idx.shape
    seg_color = np.zeros((H, W, 3), dtype=np.uint8)

    color_map = {
        0: (0, 0, 0),
        1: (128, 128, 128),
        2: (255, 0, 0),
        3: (0, 0, 255),
        4: (0, 255, 0),
    }

    for cls_id, (r, g, b) in color_map.items():
        mask = (seg_idx == cls_id)
        seg_color[mask] = [r, g, b]

    return seg_color


def save_segmentation_png_train(gt_seg, pred_seg, map_s, sample, save_dir='.', prefix='sample', mode='train'):
    if isinstance(gt_seg, torch.Tensor):
        gt_seg = gt_seg.cpu().numpy()
        if gt_seg.ndim == 4:
            gt_seg = gt_seg.squeeze(1)
    if isinstance(pred_seg, torch.Tensor):
        pred_seg = pred_seg.cpu().numpy()
    if isinstance(map_s, torch.Tensor):
        map_s = map_s.cpu().numpy()

    N, H, W = gt_seg.shape

    fig, axes = plt.subplots(nrows=N, ncols=3, figsize=(10, 5 * N))

    if N == 1:
        axes = np.expand_dims(axes, axis=0)

    if map_s.ndim == 4:
        proj_imgs = map_s[:, 1]
    else:
        raise ValueError("Projection map must be either [N, C, H, W] or [N, H, W].")

    if mode == 'train':
        label_name = 'pseudo_label'
    else:
        label_name = 'GT_label'

    for i in range(N):
        proj_img = proj_imgs[i]
        proj_color = (np.stack([proj_img] * 3, axis=-1) * 255).astype(np.uint8)

        label_img = colorize_segmentation(gt_seg[i])
        pred_img = colorize_segmentation(pred_seg[i])

        axes[i, 0].imshow(proj_color)
        axes[i, 0].set_title(f"patient_id:{sample['participant_id'][i]}, laterality:{sample['laterality'][i]}\n manufacturer:{sample['manufacturer'][i]}, anatomical:{sample['anatomical'][i]}, \n region_size:{sample['region_size'][i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label_img)
        axes[i, 1].set_title(label_name)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_img)
        axes[i, 2].set_title("model Pred")
        axes[i, 2].axis('off')

    plt.tight_layout()

    combined_save_path = osp.join(save_dir, f"{prefix}.png")
    plt.savefig(combined_save_path)
    plt.close(fig)


def adapt_epoch(args, model, optim, train_loader, test_loader, start_step, current_epoch):
    global last_checkpoint_path
    step = 0
    for sample in tqdm(train_loader, total=len(train_loader), desc="Adapting"):

        proj_map = sample['proj_map'].squeeze(1)
        imgs = sample['image'].squeeze(1)

        manufacturer_labels = torch.tensor(sample['manufacturer']).long().cuda()
        anatomical_labels = torch.tensor(sample['anatomical']).long().cuda()
        region_size_labels = torch.tensor(sample['region_size']).long().cuda()
        laterality_labels = torch.tensor(sample['laterality']).long().cuda()

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            proj_map = proj_map.cuda()

        pred, _, manufacturer_logits, anatomical_logits, region_size_logits, laterality_logits, _ = model(imgs, proj_map)

        merge_pseudo_label = sample['merge_softmax_label']

        if not isinstance(merge_pseudo_label, torch.Tensor):
            merge_pseudo_label = torch.tensor(merge_pseudo_label, device=imgs.device).long()
        else:
            merge_pseudo_label = merge_pseudo_label.to(imgs.device)

        if merge_pseudo_label.ndim == 3:
            merge_pseudo_label = merge_pseudo_label.unsqueeze(0)

        merge_pseudo_label = merge_pseudo_label.permute(0, 3, 1, 2)

        merge_pseudo_label = torch.argmax(merge_pseudo_label, dim=1)

        for i in range(anatomical_labels.shape[0]):
            if args.mask_optic_disc and anatomical_labels[i].item() == 1:
                merge_pseudo_label[i] = torch.where(merge_pseudo_label[i] == 4,
                                                    torch.zeros_like(merge_pseudo_label[i]),
                                                    merge_pseudo_label[i])

        Loss_CE_merge = torch.nn.CrossEntropyLoss()
        Loss_DSC_merge = DiceLoss()
        seg_loss = Loss_CE_merge(pred, merge_pseudo_label) + Loss_DSC_merge(pred, merge_pseudo_label)


        classification_loss_fn = torch.nn.CrossEntropyLoss()
        manufacturer_loss = classification_loss_fn(manufacturer_logits, manufacturer_labels)
        anatomical_loss = classification_loss_fn(anatomical_logits, anatomical_labels)
        region_size_loss = classification_loss_fn(region_size_logits, region_size_labels)
        laterality_loss = classification_loss_fn(laterality_logits, laterality_labels)

        aux_loss = manufacturer_loss + anatomical_loss + laterality_loss + region_size_loss


        loss = seg_loss + aux_loss

        loss.backward()
        optim.step()
        optim.zero_grad()

        if step % 200 == 0 and step > 0:
            save_dir = args.out + '/images/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pred_softmax = torch.softmax(pred, dim=1)
            pred_softmax = torch.argmax(pred, dim=1)
            save_segmentation_png_train(merge_pseudo_label, pred_softmax, proj_map, sample, save_dir=save_dir, prefix=f'training_epoch{current_epoch}_step{step}', mode='train')

        if (step % args.checkpoint_interval == 0 and step > 0):
            save_dir = args.out + '/images/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_ckpt(model, optim, args, current_epoch, step)

            if len(train_loader) - step > args.checkpoint_interval:

                csv_output_dir = osp.join(args.out, "eval")
                if not osp.exists(csv_output_dir):
                    os.makedirs(csv_output_dir)

                eval_final(args, model, test_loader, mode='')
                summary_txt_path = osp.join(args.out, f"metrics_summary_epoch{current_epoch}_step{step}.txt")
                summarize_csv_metrics(csv_output_dir, summary_txt_path)

        step += 1


def save_ckpt(model, optim, args, current_epoch=None, step=None):
    if current_epoch and step:
        global last_checkpoint_path
        checkpoint_path = osp.join(args.out, f'checkpoint_epoch{current_epoch}_step{step}.pth.tar')
        if last_checkpoint_path is not None and osp.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
    else:
        checkpoint_path = osp.join(args.out, f'last.pth.tar')
    torch.save({
        'epoch': current_epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, checkpoint_path)
    last_checkpoint_path = checkpoint_path


def main():
    now = datetime.now()
    args.out = osp.join(args.save_root, now.strftime('%Y%m%d_%H%M%S') + args.file_name)
    if not osp.exists(args.out):
        os.makedirs(args.out)
    args.out_file = open(osp.join(args.out, now.strftime('%Y%m%d')+'.txt'), 'w')
    args.out_file.write(' '.join(sys.argv) + '\n')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    args_dict = vars(args).copy()
    if "out_file" in args_dict:
        del args_dict["out_file"]

    json_path = osp.join(args.out, "args.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=2)

    custom_transform_train = Custom3DTransformTrain()
    custom_transform_weak = Custom3DTransformWeak()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_target_depth = 800

    dataset_test = AireadiSegmentation(
        root=args.data_dir,
        roi=roi_target_depth,
        device=device,
        mode='test',
        transform=custom_transform_weak,
        all_success=args.run_all_success,
        fail_image_path=args.fail_image_path,
    )

    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = [int(x) for x in args.gpu.split(',')]

    if args.method == "IPN":
        model = s_model.IPN(in_channels=args.in_channels, n_classes=args.n_classes)
    if args.method == "IPN_V2":
        model = s_model.IPNV2_with_proj_map(in_channels=args.in_channels, n_classes=args.n_classes,
                                        proj_map_in_channels=args.proj_map_channels,
                                        ava_classes=args.ava_classes, get_2D_pred=args.get_2D_pred,
                                        proj_vol_ratio=args.proj_train_ratio, dc_norms=args.dc_norms)

    if torch.cuda.is_available():
        model = model.cuda()
    log_str = '==> Loading %s model file: %s' % (model.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model.load_state_dict(checkpoint, strict=False)

    if len(device_ids) > 1:
        print("Using multiple GPUs")
        print(device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model.train()

    csv_output_dir = osp.join(args.out, "eval")
    if not osp.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    eval_final(args, model, test_loader, mode='inital')
    summary_txt_path = osp.join(args.out, f"metrics_summary_teacher_initial.txt")
    summarize_csv_metrics(csv_output_dir, summary_txt_path)

    global last_checkpoint_path
    last_checkpoint_path = None

    start_epoch = 0
    start_step = 0
    if args.resume_ckpt_path is not None:
        resume_ckpt = osp.join(args.out, args.resume_ckpt_path)
        if osp.exists(resume_ckpt):
            print("Resuming training from checkpoint:", resume_ckpt)
            checkpoint = torch.load(resume_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            print(f"Resumed from epoch {start_epoch}, step {start_step}")


    dataset_train = AireadiSegmentation(
        root=args.data_dir,
        roi=roi_target_depth,
        device=device,
        mode='train',
        transform=custom_transform_train,
        all_success=args.run_all_success,
        fail_image_path=args.fail_image_path,
    )

    g = torch.Generator()
    resume_idx = start_step * args.batch_size
    sampler = ResumeSampler(dataset_train, resume_idx=resume_idx, generator=g)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    current_epoch = 0
    sampler.set_epoch(current_epoch)

    for epoch in range(args.epoch):
        sampler.set_epoch(epoch + start_epoch +1)
        if epoch < start_epoch:
            scheduler.step()
            continue

        log_str = '\nepoch {}/{}:'.format(epoch+1, args.epoch)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        adapt_epoch(args, model, optim, train_loader, test_loader, start_step, current_epoch=epoch)

        if start_step > 0:
            start_step = 0
            sampler.set_resume_idx(0)

        scheduler.step()

        csv_output_dir = osp.join(args.out, "eval")
        if not osp.exists(csv_output_dir):
            os.makedirs(csv_output_dir)

        eval_final(args, model, test_loader, mode='')
        summary_txt_path = osp.join(args.out, f"metrics_summary_teacher_epoch{epoch}_final.txt")
        summarize_csv_metrics(csv_output_dir, summary_txt_path)

    save_ckpt(model, optim, args)


if __name__ == '__main__':
    main()

