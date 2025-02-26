import argparse
import csv
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='1')
parser.add_argument('--model-file', type=str, default='./logs_train/oneNorm/278.pth')
parser.add_argument('--file_name', type=str, default='Evaluation_image_level_model')
parser.add_argument('--model', type=str, default='IPN_V2', help='IPN_V2')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4) # Aaron lr: 0.0001
parser.add_argument('--lr-decrease-rate', type=float, default=0.9, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')

parser.add_argument('--data-dir', default='/projects/chimera/zucksliu/AI-READI-2.0/dataset/')
parser.add_argument('--dataset', type=str, default='AIREADI')
parser.add_argument('--model-source', type=str, default='OCTA500')
parser.add_argument('--batch-size', type=int, default=2)

#test155dsad
parser.add_argument('--model-ema-rate', type=float, default=0.99)
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
parser.add_argument("--gt_dir", type = str, default = "GAN_groupnorm_test_set", help="GAN_groupnorm_test_set or OneNorm_test_set")

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
import math
import torch
from torch.utils.data import DataLoader, Subset
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import torch.backends.cudnn as cudnn
import random
import sys
import json
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.metrics import dice_coefficient, assd_coefficient


# OCTA 500 import
import model
from dataloaders.aireadi_dataloader import AireadiParticipantSegmentation_2transform, AireadiParticipantSegmentation, ResumeSampler
from dataloaders.custom_octa_transform import Custom3DTransformTrain, Custom3DTransformWeak
from training_utils import DiceLoss, FocalLoss
from eval import eval_final, summarize_csv_metrics

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

# TODO: 1. Write the Patient level training

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        collated[key] = [sample[key] for sample in batch]
    return collated

def patient_collate_fn_2transform(batch):
    weak_batch = custom_collate_fn([item[0] for item in batch])
    strong_batch = custom_collate_fn([item[1] for item in batch])
    return weak_batch, strong_batch


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def save_feature_pred_bank(cavf_pred_bank, file_path):
    """
    Save the feature and prediction banks to a file.
    """
    bank = {"cavf_pred_bank": cavf_pred_bank}
    torch.save(bank, file_path)
    print(f"Saved pred bank to {file_path}")


def load_feature_pred_bank(file_path):
    """
    Load the feature and prediction banks from a file.
    Returns (feature_bank, pred_bank) if file exists; otherwise, returns (None, None).
    """
    if os.path.exists(file_path):
        bank = torch.load(file_path)
        print(f"Loaded pred bank from {file_path}")
        return bank["cavf_pred_bank"]
    else:
        return None


def init_feature_pred_bank(args, model, loader, file_path="/m-ent1/ent1/zucksliu/yuekai_DA_save/preds_bank_main.pt"):

    cavf_pred_bank = load_feature_pred_bank(file_path)
    if cavf_pred_bank is not None:
        return cavf_pred_bank, {}

    cavf_pred_bank = {}
    features_bank = {}

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        try:
            for samples in tqdm(loader, total=len(loader)):
                batch_size = len(samples["samples"])
                assert batch_size == args.batch_size
                print(samples["samples"][0][0]['image'].size())
                for i in range(batch_size):
                    print(samples["participant_id"][i])
                    for sample in samples['samples'][i]:
                        data = sample['image'].squeeze(1)
                        proj_map = sample['proj_map'].squeeze(1)
                        img_name = sample['img_name']

                        print(data.shape)

                        data = data.cuda()
                        proj_map = proj_map.cuda()

                        cavf_pred3D, _, _, _, _, _, features2D = model(data, proj_map)
                        cavf_pred3D = torch.softmax(cavf_pred3D, dim=1)

                        for j in range(data.size(0)):
                            cavf_pred_bank[img_name[j]] = cavf_pred3D[j].detach().cpu().clone()
                            features_bank[img_name[j]] = features2D[j].detach().cpu().clone()


        except Exception as e:
            print(e)
            print("Error in init_feature_pred_bank")
            # print("Saving the current feature/pred bank to file")
            # save_feature_pred_bank(cavf_pred_bank, file_path)
            exit()

    elapsed_time = time.time() - start_time
    args.out_file.write(f'Evaluation Time on Train loader weak {elapsed_time:.2f} seconds \n')
    args.out_file.flush()

    # save_feature_pred_bank(cavf_pred_bank, file_path)

    model.train()

    return cavf_pred_bank, features_bank


def soft_label_to_hard(soft_pls, pseudo_label_threshold = 0.5):
    B, C, H, W = soft_pls.shape

    y = torch.arange(H, device=soft_pls.device).float()
    x = torch.arange(W, device=soft_pls.device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0
    dist = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    threshold_map = torch.where(dist < 50, 0.2, torch.where(dist < 80, 0.7, 0.95))
    threshold_map = threshold_map.unsqueeze(0).unsqueeze(0)

    max_probs, max_idx = torch.max(soft_pls, dim=1, keepdim=True)
    hard_labels = torch.zeros_like(soft_pls)
    hard_labels.scatter_(1, max_idx, 1.0)

    cond1 = (max_idx == 1)
    cond4 = (max_idx == 4) & (max_probs >= threshold_map)
    cond_other = ((max_idx != 1) & (max_idx != 4)) & (max_probs >= pseudo_label_threshold)
    mask = torch.logical_or(cond1, torch.logical_or(cond4, cond_other)).float()

    hard_labels = hard_labels * mask

    zero_mask = (hard_labels.sum(dim=1, keepdim=True) == 0)
    hard_labels[:, 0:1] = torch.where(
        zero_mask,
        torch.ones_like(hard_labels[:, 0:1]),
        hard_labels[:, 0:1]
    )
    target_indices = torch.argmax(hard_labels, dim=1)
    return target_indices


def colorize_segmentation(seg_idx):
    """
    Convert a single-channel integer segmentation map (H,W)
    into an RGB color image (H,W,3).

    seg_idx: Torch or NumPy array of shape [H, W] with values in [0..4].
    """
    if isinstance(seg_idx, torch.Tensor):
        seg_idx = seg_idx.cpu().numpy()

    H, W = seg_idx.shape
    seg_color = np.zeros((H, W, 3), dtype=np.uint8)

    # Define color map: {class_index: (R, G, B)}
    color_map = {
        0: (0, 0, 0),         # background -> black
        1: (128, 128, 128),   # capillary  -> gray
        2: (255, 0, 0),       # artery     -> red
        3: (0, 0, 255),       # vein       -> blue
        4: (0, 255, 0),       # FAZ        -> green
    }

    for cls_id, (r, g, b) in color_map.items():
        mask = (seg_idx == cls_id)
        seg_color[mask] = [r, g, b]

    return seg_color


def save_segmentation_png(gt_seg, pred_seg, map_s, sample, dice_scores_per_class, save_dir='.', prefix='sample', mode='train'):
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
        axes[i, 2].set_title("model Pred: \n bg dice={:.4f} \n cap dice={:.4f} \n A dice={:.4f} \n V dice={:.4f} \n faz dice={:.4f}".format(dice_scores_per_class[0][-1], dice_scores_per_class[1][-1], dice_scores_per_class[2][-1], dice_scores_per_class[3][-1], dice_scores_per_class[4][-1]))
        axes[i, 2].axis('off')

    plt.tight_layout()

    combined_save_path = osp.join(save_dir, f"{prefix}.png")
    plt.savefig(combined_save_path)
    plt.close(fig)


def save_segmentation_png_qc(merge_pseudo_label, gt_seg, pred_seg, pred_seg_s, map_s, sample, save_dir='.', prefix='sample', mode='train'):
    if isinstance(gt_seg, torch.Tensor):
        gt_seg = gt_seg.cpu().numpy()
    if isinstance(pred_seg, torch.Tensor):
        pred_seg = pred_seg.cpu().numpy()
    if isinstance(map_s, torch.Tensor):
        map_s = map_s.cpu().numpy()
    if isinstance(pred_seg_s, torch.Tensor):
        pred_seg_s = pred_seg_s.cpu().numpy()

    N, H, W = gt_seg.shape

    fig, axes = plt.subplots(nrows=N, ncols=5, figsize=(10, 5 * N))

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
        pred_img_s = colorize_segmentation(pred_seg_s[i])
        merge_pseudo_label_img = colorize_segmentation(merge_pseudo_label[i])

        axes[i, 0].imshow(proj_color)
        axes[i, 0].set_title(f"patient_id:{sample['participant_id']}, laterality:{sample['laterality']}\n manufacturer:{sample['manufacturer']}, anatomical:{sample['anatomical']}, \n region_size:{sample['region_size']}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label_img)
        axes[i, 1].set_title(label_name)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_img)
        axes[i, 2].set_title("Teacher model Pred")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(pred_img_s)
        axes[i, 3].set_title("Student model Pred")
        axes[i, 3].axis('off')

        axes[i, 4].imshow(merge_pseudo_label_img)
        axes[i, 4].set_title("merge_psuedo_label")
        axes[i, 4].axis('off')

    plt.tight_layout()

    combined_save_path = osp.join(save_dir, f"{prefix}.png")
    plt.savefig(combined_save_path)
    plt.close(fig)


def adapt_epoch(args, model_t, model_s, optim, train_loader, test_loader, start_step, current_epoch, seg_loss_weight=None):
    global last_checkpoint_path
    global_step = 0
    total_steps = len(train_loader)

    w_min = 0.1
    w_max = 0.9
    try:
        for samples_w, samples_s in tqdm(train_loader, total=len(train_loader), desc="Adapting"):
            batch_size = len(samples_w["samples"])
            assert batch_size == args.batch_size
            for i in range(batch_size):

                patient_loss = 0.0
                bceloss = torch.nn.BCELoss(reduction='none')
                classification_loss_fn = torch.nn.CrossEntropyLoss()

                for k, (sample_w, sample_s) in enumerate(zip(samples_w['samples'][i], samples_s['samples'][i])):
                    imgs_w = sample_w['image'].squeeze(1)
                    map_w = sample_w['proj_map'].squeeze(1)
                    imgs_s = sample_s['image'].squeeze(1)
                    map_s = sample_s['proj_map'].squeeze(1)
                    img_name = sample_w['img_name']

                    manufacturer_labels = torch.tensor(sample_w['manufacturer']).long().cuda()
                    anatomical_labels = torch.tensor(sample_w['anatomical']).long().cuda()
                    region_size_labels = torch.tensor(sample_w['region_size']).long().cuda()
                    laterality_labels = torch.tensor(sample_w['laterality']).long().cuda()

                    manufacturer_labels = manufacturer_labels.unsqueeze(0)
                    anatomical_labels = anatomical_labels.unsqueeze(0)
                    region_size_labels = region_size_labels.unsqueeze(0)
                    laterality_labels = laterality_labels.unsqueeze(0)

                    if torch.cuda.is_available():
                        imgs_w = imgs_w.cuda()
                        imgs_s = imgs_s.cuda()
                        map_w = map_w.cuda()
                        map_s = map_s.cuda()

                    # print(k, imgs_w.shape)
                    # import time
                    # time.sleep(3)
                    # print(f"going in {k}")

                    # model predict
                    cavf_predictions_stu_s, _, manufacturer_logits_stu_s, anatomical_logits_stu_s, region_size_logits_stu_s, laterality_logits_stu_s, _ = model_s(imgs_s, map_s)
                    with torch.no_grad():
                        cavf_predictions_tea_w, _, _, _, _, _, _ = model_t(imgs_w, map_w)

                    cavf_predictions_tea_w_softmax = torch.softmax(cavf_predictions_tea_w, dim=1)
                    pseudo_labels = soft_label_to_hard(cavf_predictions_tea_w_softmax, args.pseudo_label_threshold)

                    cos_factor = (1 - math.cos(math.pi * global_step / total_steps)) / 2.0

                    teacher_faz_weight = 0.0 + (1.5 - 0.0) * cos_factor
                    merge_faz_weight = 2.0 - (2.0 - 0.5) * cos_factor

                    teacher_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, teacher_faz_weight], device='cuda')
                    merge_weights   = torch.tensor([1.0, 1.0, 1.0, 1.0, merge_faz_weight], device='cuda')

                    w_teacher_global = w_min + (w_max - w_min) * cos_factor
                    w_merge_global = 1.0 - w_teacher_global

                    merge_pseudo_label = sample_w['merge_softmax_label']

                    if not isinstance(merge_pseudo_label, torch.Tensor):
                        merge_pseudo_label = torch.tensor(merge_pseudo_label, device=imgs_s.device, dtype=torch.float32)
                    else:
                        merge_pseudo_label = merge_pseudo_label.to(imgs_s.device)

                    if merge_pseudo_label.ndim == 3:
                        merge_pseudo_label = merge_pseudo_label.unsqueeze(0)

                    merge_pseudo_label = merge_pseudo_label.permute(0, 3, 1, 2)

                    merge_pseudo_label = torch.argmax(merge_pseudo_label, dim=1)

                    if args.mask_optic_disc:
                        for i in range(anatomical_labels.shape[0]):
                            if anatomical_labels[i].item() == 1:
                                merge_pseudo_label[i] = torch.where(merge_pseudo_label[i] == 4,
                                                                    torch.zeros_like(merge_pseudo_label[i]),
                                                                    merge_pseudo_label[i])
                                pseudo_labels[i] = torch.where(pseudo_labels[i] == 4,
                                                            torch.zeros_like(pseudo_labels[i]),
                                                            pseudo_labels[i])

                    Loss_CE = torch.nn.CrossEntropyLoss(weight = teacher_weights)
                    Loss_DSC = DiceLoss(weight=teacher_weights)
                    seg_loss_teacher = Loss_CE(cavf_predictions_stu_s, pseudo_labels) + Loss_DSC(cavf_predictions_stu_s, pseudo_labels)

                    Loss_CE_merge = torch.nn.CrossEntropyLoss(weight=merge_weights)
                    Loss_DSC_merge = DiceLoss(weight=merge_weights)
                    seg_loss_merge = Loss_CE_merge(cavf_predictions_stu_s, merge_pseudo_label) + Loss_DSC_merge(cavf_predictions_stu_s, merge_pseudo_label)

                    seg_loss = w_teacher_global * seg_loss_teacher + w_merge_global * seg_loss_merge


                    classification_loss_fn = torch.nn.CrossEntropyLoss()
                    manufacturer_loss = classification_loss_fn(manufacturer_logits_stu_s, manufacturer_labels)
                    anatomical_loss = classification_loss_fn(anatomical_logits_stu_s, anatomical_labels)
                    region_size_loss = classification_loss_fn(region_size_logits_stu_s, region_size_labels)
                    laterality_loss = classification_loss_fn(laterality_logits_stu_s, laterality_labels)

                    aux_loss = manufacturer_loss + anatomical_loss + laterality_loss + region_size_loss


                    loss = seg_loss + aux_loss

                    loss.backward()

                    patient_loss += loss

                    if global_step % 100 == 0 and global_step > 0 and k == 0:
                        save_dir = args.out + '/images/'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        cavf_predictions_stu_s_seg = torch.softmax(cavf_predictions_stu_s, dim=1)
                        cavf_predictions_stu_s_seg = torch.argmax(cavf_predictions_stu_s_seg, dim=1)
                        cavf_predictions_tea_w_seg = torch.softmax(cavf_predictions_tea_w, dim=1)
                        cavf_predictions_tea_w_seg = torch.argmax(cavf_predictions_tea_w_seg, dim=1)
                        save_segmentation_png_qc(merge_pseudo_label, pseudo_labels, cavf_predictions_tea_w_seg, cavf_predictions_stu_s_seg, map_s, sample_w, save_dir=save_dir, prefix=f'training_epoch{current_epoch}_step{global_step}', mode='train')


                optim.step()
                optim.zero_grad()

                # update teacher
                for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
                    param_t.data = param_t.data.clone() * args.model_ema_rate + param_s.data.clone() * (1.0 - args.model_ema_rate)


                if global_step % args.checkpoint_interval == 0 and global_step > 0:
                    save_dir = args.out + '/images/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_ckpt(model_t, model_s, optim, args, current_epoch, global_step)

                if len(train_loader) - step > args.checkpoint_interval:
                    # metrics = eval(args, model_t, test_loader, current_epoch, step, mode='Teacher')
                    # metrics = eval(args, model_s, test_loader, current_epoch, step, mode='Student')

                    csv_output_dir = osp.join(args.out, "eval_final")
                    if not osp.exists(csv_output_dir):
                        os.makedirs(csv_output_dir)

                    eval_final(args, model_t, test_loader, mode='teacher')
                    summary_txt_path = osp.join(args.out, f"metrics_summary_teacher_epoch{current_epoch}_step{step}.txt")
                    summarize_csv_metrics(csv_output_dir, summary_txt_path)

                    eval_final(args, model_s, test_loader, mode='student')
                    summary_txt_path = osp.join(args.out, f"metrics_summary_student_epoch{current_epoch}_step{step}.txt")
                    summarize_csv_metrics(csv_output_dir, summary_txt_path)

            step += 1

    except Exception as e:
        print(e)
        save_ckpt(model_t, model_s, optim, args, current_epoch, global_step)
        # save_feature_pred_bank(cavf_pred_bank, f"/data/zucksliu/yuekai_DA_save/preds_bank_epoch{current_epoch}_{step}.pt")
        metrics = eval(args, model_t, test_loader)
        log_metrics(args, metrics, mode=f'Eval@{global_step}_teacher_model')
        exit()


def save_ckpt(model_t, model_s, optim, args, current_epoch, step):
    global last_checkpoint_path
    checkpoint_path = osp.join(args.out, f'checkpoint_epoch{current_epoch}_step{step}.pth.tar')
    if last_checkpoint_path is not None and osp.exists(last_checkpoint_path):
        os.remove(last_checkpoint_path)
    torch.save({
        'epoch': current_epoch,
        'step': step,
        'model_t_state_dict': model_t.state_dict(),
        'model_s_state_dict': model_s.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step} to {checkpoint_path}")
    last_checkpoint_path = checkpoint_path


def eval(args, model, data_loader, current_epoch=None, step=None, mode='Teacher'):
    model.eval()

    manufacturer_gt = []
    manufacturer_pred = []
    manufacturer_prob = []

    anatomical_gt = []
    anatomical_pred = []
    anatomical_prob = []

    region_size_gt = []
    region_size_pred = []
    region_size_prob = []

    laterality_gt = []
    laterality_pred = []
    laterality_prob = []

    dice_scores_per_class = {0: [], 1: [], 2: [], 3: [], 4: []}
    assd_scores_per_class = {0: [], 1: [], 2: [], 3: [], 4: []}

    if current_epoch is not None and step is not None:
        save_dir = args.out + f'/images/{mode}_Evaluation_epoch{current_epoch}_step{step}/'
    else:
        save_dir = args.out + f'/images/{mode}_Evaluation/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save_indices = set(random.sample(range(len(data_loader)), 10))
    # save_indices = set([13, 101, 119, 258, 279, 287, 308, 332, 359, 399])

    with torch.no_grad():
        for batch_idx, samples in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating"):
            batch_size = len(samples["samples"])
            assert batch_size == args.batch_size
            for i in range(batch_size):
                for idx, sample in enumerate(samples['samples'][i]):

                    data = sample['image'].squeeze(1)
                    proj_map = sample['proj_map'].squeeze(1)

                    manufacturer_labels = torch.tensor(sample['manufacturer']).long().cuda()
                    anatomical_labels = torch.tensor(sample['anatomical']).long().cuda()
                    region_size_labels = torch.tensor(sample['region_size']).long().cuda()
                    laterality_labels = torch.tensor(sample['laterality']).long().cuda()

                    data = data.cuda()
                    proj_map = proj_map.cuda()

                    cavf_pred3D, _, manufacturer_logits, anatomical_logits, region_size_logits, laterality_logits, _ = model(data, proj_map)
                    cavf_pred3D = torch.softmax(cavf_pred3D, dim=1)

                    gt_seg = sample['data_label']
                    if isinstance(gt_seg, np.ndarray):
                        gt_seg = torch.from_numpy(gt_seg).to(data.device)

                    if gt_seg.dim() == 3:
                        gt_seg = gt_seg.unsqueeze(0)

                    if gt_seg.dim() == 4 and gt_seg.shape[1] == 5:
                        gt_seg = torch.argmax(gt_seg, dim=1)
                    pred_seg = torch.argmax(cavf_pred3D, dim=1)

                    for i in range(anatomical_labels.shape[0]):
                        if args.mask_optic_disc and anatomical_labels[i].item() == 1:
                            gt_seg[i] = torch.where(gt_seg[i] == 4,
                                                                torch.zeros_like(gt_seg[i]),
                                                                gt_seg[i])


                    # if batch_idx in save_indices:


                    for i in range(pred_seg.size(0)):
                        for cls in range(5):
                            pred_mask = (pred_seg[i] == cls).float()
                            gt_mask   = (gt_seg[i]   == cls).float()
                            dice = dice_coefficient(pred_mask, gt_mask)
                            dice_scores_per_class[cls].append(dice)

                            pred_mask_np = (pred_mask.cpu().numpy() > 0.5).astype(np.bool_)
                            gt_mask_np   = (gt_mask.cpu().numpy() > 0.5).astype(np.bool_)
                            assd = assd_coefficient(pred_mask_np, gt_mask_np)
                            assd_scores_per_class[cls].append(assd)

                    save_segmentation_png(gt_seg, pred_seg, proj_map, sample, dice_scores_per_class, save_dir=save_dir, prefix=f'Evaluation_batch_idx{batch_idx}_{i}_{idx}', mode='eval')

                    manufacturer_probs = torch.softmax(manufacturer_logits, dim=1)
                    anatomical_probs   = torch.softmax(anatomical_logits, dim=1)
                    region_size_probs  = torch.softmax(region_size_logits, dim=1)
                    laterality_probs   = torch.softmax(laterality_logits, dim=1)

                    manufacturer_preds = torch.argmax(manufacturer_probs, dim=1)
                    anatomical_preds   = torch.argmax(anatomical_probs, dim=1)
                    region_size_preds  = torch.argmax(region_size_probs, dim=1)
                    laterality_preds   = torch.argmax(laterality_probs, dim=1)

                    manufacturer_gt.append(manufacturer_labels.cpu().numpy())
                    manufacturer_pred.append(manufacturer_preds.cpu().numpy())
                    manufacturer_prob.append(manufacturer_probs.cpu().numpy())

                    anatomical_gt.append(anatomical_labels.cpu().numpy())
                    anatomical_pred.append(anatomical_preds.cpu().numpy())
                    anatomical_prob.append(anatomical_probs.cpu().numpy())

                    region_size_gt.append(region_size_labels.cpu().numpy())
                    region_size_pred.append(region_size_preds.cpu().numpy())
                    region_size_prob.append(region_size_probs.cpu().numpy())

                    laterality_gt.append(laterality_labels.cpu().numpy())
                    laterality_pred.append(laterality_preds.cpu().numpy())
                    laterality_prob.append(laterality_probs.cpu().numpy())

    manufacturer_gt = np.array(manufacturer_gt)
    manufacturer_pred = np.array(manufacturer_pred)
    manufacturer_prob = np.array(manufacturer_prob)

    anatomical_gt = np.array(anatomical_gt)
    anatomical_pred = np.array(anatomical_pred)
    anatomical_prob = np.array(anatomical_prob)

    region_size_gt = np.array(region_size_gt)
    region_size_pred = np.array(region_size_pred)
    region_size_prob = np.array(region_size_prob)

    laterality_gt = np.array(laterality_gt)
    laterality_pred = np.array(laterality_pred)
    laterality_prob = np.array(laterality_prob)

    metrics = {}

    metrics['manufacturer_accuracy'] = accuracy_score(manufacturer_gt, manufacturer_pred)
    metrics['manufacturer_f1'] = f1_score(manufacturer_gt, manufacturer_pred, average='macro')
    try:
        metrics['manufacturer_auc'] = roc_auc_score(manufacturer_gt, manufacturer_prob, multi_class='ovr')
    except Exception as e:
        metrics['manufacturer_auc'] = np.nan
        print("Manufacturer AUC calculation error:", e)

    metrics['anatomical_accuracy'] = accuracy_score(anatomical_gt, anatomical_pred)
    metrics['anatomical_f1'] = f1_score(anatomical_gt, anatomical_pred, average='macro')
    try:
        metrics['anatomical_auc'] = roc_auc_score(anatomical_gt, anatomical_prob, multi_class='ovr')
    except Exception as e:
        metrics['anatomical_auc'] = np.nan
        print("Anatomical AUC calculation error:", e)

    metrics['region_size_accuracy'] = accuracy_score(region_size_gt, region_size_pred)
    metrics['region_size_f1'] = f1_score(region_size_gt, region_size_pred, average='macro')
    try:
        metrics['region_size_auc'] = roc_auc_score(region_size_gt, region_size_prob, multi_class='ovr')
    except Exception as e:
        metrics['region_size_auc'] = np.nan
        print("Region Size AUC calculation error:", e)

    metrics['laterality_accuracy'] = accuracy_score(laterality_gt, laterality_pred)
    metrics['laterality_f1'] = f1_score(laterality_gt, laterality_pred, average='macro')
    try:
        metrics['laterality_auc'] = roc_auc_score(laterality_gt, laterality_prob, multi_class='ovr')
    except Exception as e:
        metrics['laterality_auc'] = np.nan
        print("Laterality AUC calculation error:", e)

    dice_mean = {}
    dice_std = {}
    assd_mean = {}
    assd_std = {}
    for cls in range(5):
        if len(dice_scores_per_class[cls]) > 0:
            dice_mean[cls] = np.mean(dice_scores_per_class[cls])
            dice_std[cls] = np.std(dice_scores_per_class[cls])
        else:
            dice_mean[cls] = float('nan')
            dice_std[cls] = float('nan')
        if len(assd_scores_per_class[cls]) > 0:
            assd_mean[cls] = np.mean(assd_scores_per_class[cls])
            assd_std[cls] = np.std(assd_scores_per_class[cls])
        else:
            assd_mean[cls] = float('nan')
            assd_std[cls] = float('nan')

    metrics['segmentation_dice_mean'] = dice_mean
    metrics['segmentation_dice_std'] = dice_std
    metrics['segmentation_assd_mean'] = assd_mean
    metrics['segmentation_assd_std'] = assd_std

    seg_dice_str = (
        "Segmentation Dice coefficients (mean ± std):\n"
        "  Background: {:.4f} ± {:.4f}\n"
        "  Capillary : {:.4f} ± {:.4f}\n"
        "  Artery    : {:.4f} ± {:.4f}\n"
        "  Vein      : {:.4f} ± {:.4f}\n"
        "  FAZ       : {:.4f} ± {:.4f}\n"
        .format(
            dice_mean[0], dice_std[0],
            dice_mean[1], dice_std[1],
            dice_mean[2], dice_std[2],
            dice_mean[3], dice_std[3],
            dice_mean[4], dice_std[4]
        )
    )
    seg_assd_str = (
        "Segmentation ASSD (mm) (mean ± std):\n"
        "  Background: {:.4f} ± {:.4f}\n"
        "  Capillary : {:.4f} ± {:.4f}\n"
        "  Artery    : {:.4f} ± {:.4f}\n"
        "  Vein      : {:.4f} ± {:.4f}\n"
        "  FAZ       : {:.4f} ± {:.4f}\n"
        .format(
            assd_mean[0], assd_std[0],
            assd_mean[1], assd_std[1],
            assd_mean[2], assd_std[2],
            assd_mean[3], assd_std[3],
            assd_mean[4], assd_std[4]
        )
    )
    metrics['segmentation_dice_str'] = seg_dice_str
    metrics['segmentation_assd_str'] = seg_assd_str

    model.train()
    return metrics


def log_metrics(args, metrics, mode='Initial'):
    log_str = (
        f"{mode} Extra Label Metrics:\n"
        f"  Manufacturer: Accuracy: {metrics['manufacturer_accuracy']:.4f}, F1: {metrics['manufacturer_f1']:.4f}, AUC: {metrics['manufacturer_auc']:.4f}\n"
        f"  Anatomical  : Accuracy: {metrics['anatomical_accuracy']:.4f}, F1: {metrics['anatomical_f1']:.4f}, AUC: {metrics['anatomical_auc']:.4f}\n"
        f"  Region Size : Accuracy: {metrics['region_size_accuracy']:.4f}, F1: {metrics['region_size_f1']:.4f}, AUC: {metrics['region_size_auc']:.4f}\n"
        f"  Laterality  : Accuracy: {metrics['laterality_accuracy']:.4f}, F1: {metrics['laterality_f1']:.4f}, AUC: {metrics['laterality_auc']:.4f}\n"
    )
    log_str += "\n" + metrics['segmentation_dice_str'] + "\n"
    log_str += metrics['segmentation_assd_str'] + "\n"

    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()


def main():
    now = datetime.now()
    args.out = osp.join('/m-ent1/ent1/zucksliu/SFDA-CBMT_results', now.strftime('%Y%m%d_%H%M%S') + args.file_name)
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

    dataset_test = AireadiParticipantSegmentation(
        root=args.data_dir,
        roi=roi_target_depth,
        device=device,
        mode='test',
        transform=custom_transform_weak,
        label_dir=args.gt_dir,
        all_success=args.run_all_success
    )

    # train_loader_weak = DataLoader(dataset_train_weak, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = [int(x) for x in args.gpu.split(',')]

    if args.method == "IPN":
        model_s = model.IPN(in_channels=args.in_channels, n_classes=args.n_classes)
        model_t = model.IPN(in_channels=args.in_channels, n_classes=args.n_classes)
    if args.method == "IPN_V2":
        model_s = model.IPNV2_with_proj_map(in_channels=args.in_channels, n_classes=args.n_classes,
                                        proj_map_in_channels=args.proj_map_channels,
                                        ava_classes=args.ava_classes, get_2D_pred=args.get_2D_pred,
                                        proj_vol_ratio=args.proj_train_ratio, dc_norms=args.dc_norms)
        model_t = model.IPNV2_with_proj_map(in_channels=args.in_channels, n_classes=args.n_classes,
                                        proj_map_in_channels=args.proj_map_channels,
                                        ava_classes=args.ava_classes, get_2D_pred=args.get_2D_pred,
                                        proj_vol_ratio=args.proj_train_ratio, dc_norms=args.dc_norms)

    if torch.cuda.is_available():
        model_s = model_s.cuda()
        model_t = model_t.cuda()
    log_str = '==> Loading %s model file: %s' % (model_s.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model_s.load_state_dict(checkpoint, strict=False)
    model_t.load_state_dict(checkpoint, strict=False)

    if len(device_ids) > 1:
        print("Using multiple GPUs")
        print(device_ids)
        model_s = torch.nn.DataParallel(model_s, device_ids=device_ids)
        model_t = torch.nn.DataParallel(model_t, device_ids=device_ids)

    optim = torch.optim.Adam(model_s.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model_s.train()
    model_t.train()
    for param in model_t.parameters():
        param.requires_grad = False

    # cavf_pred_bank, features_bank = init_feature_pred_bank(args, model_s, train_loader_weak, file_path="/m-ent1/ent1/zucksliu/yuekai_DA_save/preds_bank_main.pt")

    global last_checkpoint_path
    last_checkpoint_path = None

    start_epoch = 0
    start_step = 0
    if args.resume_ckpt_path is not None:
        resume_ckpt = osp.join(args.out, args.resume_ckpt_path)
        if osp.exists(resume_ckpt):
            print("Resuming training from checkpoint:", resume_ckpt)
            checkpoint = torch.load(resume_ckpt)
            model_s.load_state_dict(checkpoint['model_state_dict'])
            model_t.load_state_dict(checkpoint['model_state_dict'])
            # optim.load_state_dict(checkpoint['optimizer_state_dict'])
            # start_epoch = checkpoint['epoch']
            # start_step = checkpoint['step']
            # print(f"Resumed from epoch {start_epoch}, step {start_step}")


    # metrics = eval(args, model_t, test_loader, mode='Initial')
    # log_metrics(args, metrics, mode='Initial')


    dataset_train = AireadiParticipantSegmentation_2transform(
        root=args.data_dir,
        roi=roi_target_depth,
        device=device,
        mode='train',
        transform_strong=custom_transform_train,
        transform_weak=custom_transform_weak,
        label_dir=args.gt_dir,
        all_success=args.run_all_success
    )

    g = torch.Generator()
    resume_idx = start_step * args.batch_size
    sampler = ResumeSampler(dataset_train, resume_idx=resume_idx, generator=g)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler, num_workers=4, collate_fn=patient_collate_fn_2transform)

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

        adapt_epoch(args, model_t, model_s, optim, train_loader, test_loader, start_step, current_epoch=epoch)

        if start_step > 0:
            start_step = 0
            sampler.set_resume_idx(0)

        scheduler.step()

        # metrics = eval(args, model_t, test_loader, current_epoch=epoch, step=len(train_loader), mode='Teacher')
        # log_metrics(args, metrics, mode='Teacher')

        # metrics = eval(args, model_s, test_loader, current_epoch=epoch, step=len(train_loader), mode='Student')
        # log_metrics(args, metrics, mode='Student')

        csv_output_dir = osp.join(args.out, "eval_final")
        if not osp.exists(csv_output_dir):
            os.makedirs(csv_output_dir)

        eval_final(args, model_t, test_loader, mode='teacher')
        summary_txt_path = osp.join(args.out, f"metrics_summary_teacher_epoch{current_epoch}_final.txt")
        summarize_csv_metrics(csv_output_dir, summary_txt_path)
        eval_final(args, model_s, test_loader, mode='student')
        summary_txt_path = osp.join(args.out, f"metrics_summary_student_epoch{current_epoch}_final.txt")
        summarize_csv_metrics(csv_output_dir, summary_txt_path)

    torch.save({'model_state_dict': model_t.state_dict()}, args.out + '/last.pth.tar')


if __name__ == '__main__':
    main()

