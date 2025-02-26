# Developed by Yuekai Xu, Aaron Honjaya, Zixuan Liu, all rights reserved to GrInAdapt team.

import argparse
import os
import os.path as osp
import torch.nn.functional as F
import torch.nn as nn

import torch
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import cv2
import torch.backends.cudnn as cudnn
import random
from utils.metrics import *
import os.path as osp

import model_with_dropout as model
from dataloaders.aireadi_dataloader import AireadiSegmentation, AireadiSegmentation_2transform, ResumeSampler
from dataloaders.custom_octa_transform import Custom3DTransformTrain, Custom3DTransformWeak

bceloss = torch.nn.BCELoss()
seed = 3377
savefig = False
get_hd = False
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

def disenable_dropout(model):
    """ Function to disenable the dropout layers during test-time """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.eval()



def soft_label_to_hard(soft_pls, pseudo_label_threshold = 0.5):
    B, C, H, W = soft_pls.shape

    bg_prob = soft_pls[:, 0:1, :, :]
    faz_prob = soft_pls[:, 1:2, :, :]

    y = torch.arange(H, device=soft_pls.device).float()
    x = torch.arange(W, device=soft_pls.device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0
    dist = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    threshold_map = torch.where(dist < 60, 0.5, torch.where(dist < 80, 0.5, 0.5))
    threshold_map = threshold_map.unsqueeze(0).unsqueeze(0)

    faz_highest = (faz_prob >= bg_prob).float()

    pseudo_labels_obj = ((faz_prob >= threshold_map) * faz_highest).long()

    pseudo_labels_bg = 1 - pseudo_labels_obj

    pseudo_labels = torch.cat([pseudo_labels_bg, pseudo_labels_obj], dim=1)
    return pseudo_labels


def create_color_composite(two_channel_map):
    H, W = two_channel_map.shape[1:]
    bg_color = np.array([0, 0, 128], dtype=np.float32)
    fg_color = np.array([0, 255, 0], dtype=np.float32)

    bg_prob = two_channel_map[0, :, :].reshape(H, W, 1)
    fg_prob = two_channel_map[1, :, :].reshape(H, W, 1)

    composite = bg_prob * bg_color + fg_prob * fg_color

    composite = np.clip(composite, 0, 255).astype(np.uint8)

    return composite


def retain_largest_cluster_in_circle(proto_pseudo, radius=80):
    if isinstance(proto_pseudo, torch.Tensor):
        proto_pseudo = proto_pseudo.cpu().numpy()

    if proto_pseudo.ndim == 4:
        if proto_pseudo.shape[0] > 1:
            print("Warning: Batch size > 1. Using the first sample only.")
        proto_pseudo = proto_pseudo[0]

    if proto_pseudo.ndim == 2:
        fg = proto_pseudo.copy()
        bg = 1 - fg
        proto_pseudo = np.stack([bg, fg], axis=0)
    elif proto_pseudo.shape[0] == 1:
        fg = proto_pseudo[0]
        bg = 1 - fg
        proto_pseudo = np.stack([bg, fg], axis=0)

    fg_mask = np.squeeze(proto_pseudo[1]).astype(np.uint8)
    H, W = fg_mask.shape

    center = (W // 2, H // 2)
    circle_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 1, -1)

    fg_mask_circle = cv2.bitwise_and(fg_mask, fg_mask, mask=circle_mask)
    fg_mask_circle = fg_mask_circle.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask_circle, connectivity=8)

    if num_labels <= 1:
        return proto_pseudo[None, ...]

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    new_fg_mask = (labels == largest_label).astype(np.uint8)

    new_proto = np.empty((2, H, W), dtype=proto_pseudo.dtype)
    new_proto[1] = new_fg_mask
    new_proto[0] = 1 - new_fg_mask

    new_proto = new_proto[None, ...]

    return new_proto


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./models/oneNorm/278.pth')
    parser.add_argument('--dataset', type=str, default='AIREADI')#Domain1
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--source', type=str, default='OCTA500')#Domain4
    parser.add_argument('--save_root', type=str, default='./log_results/')
    parser.add_argument('-g', '--gpu', type=str, default='3')
    parser.add_argument('--data-dir', default='/projects/chimera/zucksliu/AI-READI-2.0/dataset/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--save-root-ent',type=str,default='./results/ent/')
    parser.add_argument('--save-root-mask',type=str,default='./results/mask/')
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--test-prediction-save-path', type=str,default='./results/baseline/')

    parser.add_argument("--in_channels", type=int, default=2, help="input channels")
    parser.add_argument("--n_classes", type=int, default=5, help="class number")
    parser.add_argument("--method", type=str, default="IPN_V2", help="IPN, IPN_V2")
    parser.add_argument("--ava_classes", type=int, default=2, help="label channels")
    parser.add_argument("--proj_map_channels", type=int, default=2, help="class number")
    parser.add_argument("--get_2D_pred", type=bool, default=True, help="get 2D head")
    parser.add_argument("--proj_train_ratio", type=int, default=1, help="proj_map H or W to train_size H or W ratio. Currently only supports 1 or 2")
    parser.add_argument("--dc_norms", type = str, default = "NG", help="normalization for Double Conv")
    parser.add_argument("--gt_dir", type = str, default = "OneNorm_test_set", help="GAN_groupnorm_test_set or OneNorm_test_set")

    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Save model checkpoint every K patient updates')
    parser.add_argument('--resume-ckpt-path', type=str, default=None, help='Path to resume checkpoint')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 2. model
    model = model.IPNV2_with_proj_map(in_channels=args.in_channels, n_classes=args.n_classes,
                                        proj_map_in_channels=args.proj_map_channels,
                                        ava_classes=args.ava_classes, get_2D_pred=args.get_2D_pred,
                                        proj_vol_ratio=args.proj_train_ratio, dc_norms=args.dc_norms)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(args.model_file)
    model.load_state_dict(checkpoint, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_target_depth = 800

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = [int(x) for x in args.gpu.split(',')]

    # 1. dataset
    custom_transform_test = Custom3DTransformWeak()

    dataset_train = AireadiSegmentation(
        root=args.data_dir,
        roi=roi_target_depth,
        device=device,
        mode='train',
        transform=custom_transform_test,
        label_dir=args.gt_dir
    )

    g = torch.Generator()
    resume_idx = 0
    sampler = ResumeSampler(dataset_train, resume_idx=resume_idx, generator=g)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    model.eval()
    enable_dropout(model)

    if len(device_ids) > 1:
        print("Using multiple GPUs")
        print(device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    pseudo_label_dic = {}
    uncertain_dic = {}
    proto_pseudo_dic = {}
    prob_dic = {}

    save_dir = args.save_root
    save_image_dir = os.path.join(save_dir, 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)


    with torch.no_grad():
        for batch_idx, sample in tqdm.tqdm(enumerate(train_loader),
                                             total=len(train_loader)):

            data = sample['image'].squeeze(1)
            img_name = sample['img_name']
            proj_map = sample['proj_map'].squeeze(1)

            if torch.cuda.is_available():
                data = data.cuda()
                proj_map = proj_map.cuda()

            num_ensembles = 10
            preds = torch.zeros([num_ensembles, data.shape[0], 2, data.shape[3], data.shape[4]]).cuda()
            features = torch.zeros([num_ensembles, data.shape[0], 128, 256, 256]).cuda()

            for i in range(num_ensembles):
                with torch.no_grad():
                    out, _, _, _, _, _, feat, _ = model(data, proj_map)
                    fg_logits = out[:, 4:5, ...]
                    bg_logits = out[:, :4, ...].mean(dim=1, keepdim=True)
                    fg_exp = torch.exp(fg_logits)
                    bg_exp = torch.exp(bg_logits)
                    total = fg_exp + bg_exp + 1e-8
                    fg_prob = fg_exp / total
                    bg_prob = bg_exp / total

                    out_2ch = torch.cat([bg_prob, fg_prob], dim=1)
                    preds[i, ...] = out_2ch
                    features[i, ...] = feat

            preds = preds.cpu()
            features = features.cpu()

            preds1 = torch.sigmoid(preds)
            preds = torch.sigmoid(preds/2.0)
            std_map = torch.std(preds,dim=0)

            prediction=torch.mean(preds1,dim=0)
            prob = prediction.clone()

            pseudo_label = soft_label_to_hard(prediction, pseudo_label_threshold=0.5)

            feature = torch.mean(features,dim=0)

            target_0_obj = F.interpolate(pseudo_label[:,0:1,...].float(), size=feature.size()[2:], mode='nearest')
            target_1_obj = F.interpolate(pseudo_label[:, 1:, ...].float(), size=feature.size()[2:], mode='nearest')
            prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)
            std_map_small = F.interpolate(std_map, size=feature.size()[2:], mode='bilinear', align_corners=True)
            target_0_bck = 1.0 - target_0_obj
            target_1_bck = 1.0 - target_1_obj

            mask_0_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]])
            mask_0_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]])
            mask_1_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]])
            mask_1_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]])
            mask_0_obj[std_map_small[:, 0:1, ...] < 0.1] = 1.0
            mask_0_bck[std_map_small[:, 0:1, ...] < 0.1] = 1.0
            mask_1_obj[std_map_small[:, 1:, ...] < 0.1] = 1.0
            mask_1_bck[std_map_small[:, 1:, ...] < 0.1] = 1.0
            mask_0 = mask_0_obj + mask_0_bck
            mask_1 = mask_1_obj + mask_1_bck
            mask = torch.cat((mask_0, mask_1), dim=1)

            feature_0_obj = feature * target_0_obj*mask_0_obj # class 0 with high confidence
            feature_1_obj = feature * target_1_obj*mask_1_obj # class 1 with high confidence
            feature_0_bck = feature * target_0_bck*mask_0_bck # class 1 with high confidence shown in class 0
            feature_1_bck = feature * target_1_bck*mask_1_bck # class 0 with high confidence shown in class 1

            # prediction_small is the probability map
            # The below 4 lines get the sum of the feature that has high confidence in class 0 and class 1
            # The following 12 lines is getting the mean (centroid) of the feature that has high confidence in class 0 and class 1
            centroid_0_obj = torch.sum(feature_0_obj*prediction_small[:,0:1,...], dim=[0,2,3], keepdim=True)
            centroid_1_obj = torch.sum(feature_1_obj*prediction_small[:,1:,...], dim=[0,2,3], keepdim=True)
            centroid_0_bck = torch.sum(feature_0_bck*(1.0-prediction_small[:,0:1,...]), dim=[0,2,3], keepdim=True)
            centroid_1_bck = torch.sum(feature_1_bck*(1.0-prediction_small[:,1:,...]), dim=[0,2,3], keepdim=True)

            # mask_i_{} * target_i_{} is the pixels with high confidence in the corresponding class from the obj/bck
            target_0_obj_cnt = torch.sum(mask_0_obj*target_0_obj*prediction_small[:,0:1,...], dim=[0,2,3], keepdim=True)
            target_1_obj_cnt = torch.sum(mask_1_obj*target_1_obj*prediction_small[:,1:,...], dim=[0,2,3], keepdim=True)
            target_0_bck_cnt = torch.sum(mask_0_bck*target_0_bck*(1.0-prediction_small[:,0:1,...]), dim=[0,2,3], keepdim=True)
            target_1_bck_cnt = torch.sum(mask_1_bck*target_1_bck*(1.0-prediction_small[:,1:,...]), dim=[0,2,3], keepdim=True)

            centroid_0_obj /= target_0_obj_cnt
            centroid_1_obj /= target_1_obj_cnt
            centroid_0_bck /= target_0_bck_cnt
            centroid_1_bck /= target_1_bck_cnt

            distance_0_obj = torch.sum(torch.pow(feature - centroid_0_obj, 2), dim=1, keepdim=True)
            distance_0_bck = torch.sum(torch.pow(feature - centroid_0_bck, 2), dim=1, keepdim=True)
            distance_1_obj = torch.sum(torch.pow(feature - centroid_1_obj, 2), dim=1, keepdim=True)
            distance_1_bck = torch.sum(torch.pow(feature - centroid_1_bck, 2), dim=1, keepdim=True)

            proto_pseudo_0 = torch.zeros([data.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
            proto_pseudo_1 = torch.zeros([data.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()

            proto_pseudo_0[distance_0_obj < distance_0_bck] = 1.0
            proto_pseudo_1[distance_1_obj < distance_1_bck] = 1.0
            proto_pseudo = torch.cat((proto_pseudo_0, proto_pseudo_1), dim=1)
            proto_pseudo = F.interpolate(proto_pseudo, size=data.size()[3:], mode='nearest')

            debugc = 1

            pseudo_label = pseudo_label.detach().cpu().numpy()
            std_map = std_map.detach().cpu().numpy()
            proto_pseudo = proto_pseudo.detach().cpu().numpy()
            prob = prob.detach().cpu().numpy()
            for i in range(prediction.shape[0]): # default fea_channels=128
                pseudo_label_dic[img_name[i]] = pseudo_label[i] # simple pseudo label, shape: [2, H, W]
                uncertain_dic[img_name[i]] = std_map[i] # uncertainty map, shape: [2, H, W]
                proto_pseudo_dic[img_name[i]] = proto_pseudo[i] # prototypical pseudo label, shape: [2, H, W]
                prob_dic[img_name[i]] = prob[i] # probability map, range [0,1], shape: [2, H, W]

                proto_pseudo_np = proto_pseudo[i]  # shape: [2, H, W]
                pseudo_label_np = pseudo_label[i]  # shape: [2, H, W]
                uncertain_np    = std_map[i]        # shape: [2, H, W]

                proto_color   = create_color_composite(proto_pseudo_np)
                pseudo_color  = create_color_composite(pseudo_label_np)
                proj_map_np = proj_map[0, 1].cpu().numpy()

                uncertain_np = uncertain_np[i]

                uncertain_norm = uncertain_np

                fig, axes = plt.subplots(1, 4, figsize=(20, 8))

                axes[0].imshow(proj_map_np, cmap='gray')
                axes[0].set_title(f"patient_id:{sample['participant_id'][i]}, laterality:{sample['laterality'][i]}\n manufacturer:{sample['manufacturer'][i]}, anatomical:{sample['anatomical'][i]}, \n region_size:{sample['region_size'][i]}")
                axes[0].axis("off")

                # Plot the prototypical pseudo label composite
                axes[1].imshow(proto_color)
                axes[1].set_title("Proto Pseudo")
                axes[1].axis("off")

                # Plot the pseudo label composite
                axes[2].imshow(pseudo_color)
                axes[2].set_title("Pseudo Label")
                axes[2].axis("off")

                # Plot the thresholded uncertainty composite
                axes[3].imshow(uncertain_norm, cmap='gray')
                axes[3].set_title("Uncertainty")
                axes[3].axis("off")

                plt.tight_layout()
                # save_image_path = os.path.join(save_image_dir, f"{img_name[i]}.png")
                save_image_path = os.path.join(save_image_dir, "test.png")
                plt.savefig(save_image_path)
                plt.close()


    save_path = os.path.join(save_dir, 'pseudolabels.npz')
    np.savez(save_path, pseudo_label_dic, uncertain_dic, proto_pseudo_dic, prob_dic)

