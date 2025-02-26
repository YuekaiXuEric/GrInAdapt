import torch
import numpy as np
import medpy.metric.binary as medmetric
from scipy.ndimage import binary_erosion, distance_transform_edt

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1,i+1):
            x = fi+j*j
            if f[i-j] < x: break
            f[i-j] = x


def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    # segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    segmentation_pixels = np.sum(binary_segmentation.astype(float), axis=(1,2))
    # same for the ground truth
    # gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    gt_label_pixels = np.sum(binary_gt_label.astype(float), axis=(1,2))
    # same for the intersection
    intersection = np.sum(intersection.astype(float), axis=(1,2))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def dice_coefficient_numpy_3D(binary_segmentation, binary_gt_label):
    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    # segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    segmentation_pixels = np.sum(binary_segmentation.astype(float), axis=(0,1,2))
    # same for the ground truth
    # gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    gt_label_pixels = np.sum(binary_gt_label.astype(float), axis=(0,1,2))
    # same for the intersection
    intersection = np.sum(intersection.astype(float), axis=(0,1,2))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def dice_numpy_medpy(binary_segmentation, binary_gt_label):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)

    return medmetric.dc(binary_segmentation, binary_gt_label)


def assd_numpy(binary_segmentation, binary_gt_label):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)

    if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
        return medmetric.assd(binary_segmentation, binary_gt_label)
    else:
        return -1


def hd_numpy(binary_segmentation, binary_gt_label):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)

    if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
        return medmetric.hd(binary_segmentation, binary_gt_label)
    else:
        return -1


def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return dice_coefficient_numpy(pred, target)

def dice_coeff_2label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0
    return dice_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]), dice_coefficient_numpy(pred[:, 1, ...],
                                                                                              target[:, 1, ...])


def dice_coeff_4label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    y_hat = torch.argmax(pred, dim=1, keepdim=True)
    pred_label = torch.zeros(pred.size())  # bs*4*W*H
    if torch.cuda.is_available():
        pred_label = pred_label.cuda()
    pred_label = pred_label.scatter_(1, y_hat, 1)  # one-hot label
    target = target.data.cpu()
    pred_label = pred_label.data.cpu()
    return (dice_coefficient_numpy(pred_label[:, 0, ...], target[:, 0, ...]),
            dice_coefficient_numpy(pred_label[:, 1, ...], target[:, 1, ...]),
            dice_coefficient_numpy(pred_label[:, 2, ...], target[:, 2, ...]),
            dice_coefficient_numpy(pred_label[:, 3, ...], target[:, 3, ...]))


def DiceLoss(input, target):
    '''
    in tensor fomate
    :param input:
    :param target:
    :return:
    '''
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def assd_compute(pred, target):
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0

    assd = np.zeros([pred.shape[0], pred.shape[1]])
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            assd[i][j] = assd_numpy(pred[i, j, ...], target[i, j, ...])

    return assd


def dice_coeff_5label(pred, target, threshold=0.75):
    """
    Compute Dice coefficients for 5 classes.

    Args:
        pred: a torch tensor of shape (N, 5, H, W) (logits)
        target: a torch tensor of shape (N, 5, H, W) (one-hot binary labels)
        threshold: threshold to binarize the predictions.

    Returns:
        A tuple of 5 values (one for each class) representing the average Dice
        coefficient computed over the batch.
    """
    # Bring to CPU and apply sigmoid on predictions
    target = target.data.cpu()
    pred = torch.sigmoid(pred).data.cpu()
    # Binarize predictions
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    N = pred.shape[0]
    dice_list = []
    for c in range(5):
        # Extract channel c from predictions and targets.
        pred_c = pred[:, c, :, :]  # shape (N, H, W)
        target_c = target[:, c, :, :]  # shape (N, H, W)
        dice_values = dice_coefficient_numpy(pred_c, target_c)  # returns an array of shape (N,)
        # Take the mean over the batch
        dice_mean = np.mean(dice_values)
        dice_list.append(dice_mean)
    return tuple(dice_list)

def assd_compute_5label(pred, target, threshold=0.75):
    """
    Compute the Average Symmetric Surface Distance (ASSD) for 5 classes.

    Args:
        pred: a torch tensor of shape (N, 5, H, W) (logits)
        target: a torch tensor of shape (N, 5, H, W) (one-hot binary labels)
        threshold: threshold to binarize the predictions.

    Returns:
        A NumPy array of shape (N, 5) where each element [n, c] is the ASSD for sample n, class c.
    """
    target = target.data.cpu()
    pred = torch.sigmoid(pred).data.cpu()
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    N = pred.shape[0]
    assd_arr = np.zeros((N, 5))
    for n in range(N):
        for c in range(5):
            # Compute ASSD for sample n, class c.
            assd_arr[n, c] = assd_numpy(pred[n, c, :, :], target[n, c, :, :])
    return assd_arr


def global_avg_pool(logits):
    if logits.dim() > 2:
        return logits.mean(dim=[2, 3])
    else:
        return logits


def dice_coefficient(pred, target, smooth=1e-6):
    target = target.to(pred.device)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def assd_coefficient(pred_mask, gt_mask):
    pred_boundary = pred_mask & (~binary_erosion(pred_mask))
    gt_boundary   = gt_mask   & (~binary_erosion(gt_mask))

    if not np.any(pred_boundary) or not np.any(gt_boundary):
        return np.nan

    dt_gt = distance_transform_edt(~gt_boundary)
    dt_pred = distance_transform_edt(~pred_boundary)

    dt_gt = dt_gt.squeeze()  # Remove singleton dimensions
    pred_boundary = pred_boundary.squeeze()
    gt_boundary = gt_boundary.squeeze()

    pred_to_gt = dt_gt[pred_boundary]
    gt_to_pred = dt_pred[gt_boundary]

    all_distances = np.concatenate([pred_to_gt, gt_to_pred])
    return all_distances.mean()