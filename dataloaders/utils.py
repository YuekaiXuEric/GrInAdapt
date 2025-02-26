import os
import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from PIL import Image

def check_dir_exist(dir):
    """create directories"""
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ""
        for name in names:
            dir = os.path.join(dir, name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print("dir", "'" + dir + "'", "is created.")


def cal_Dice(img1, img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i, j] >= 1 and img2[i, j] >= 1:
                I += 1
            if img1[i, j] >= 1 or img2[i, j] >= 1:
                U += 1
    return 2 * I / (I + U + 1e-5)


def cal_acc(img1, img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i, j] == img2[i, j]:
                acc += 1
    return acc / (shape[0] * shape[1])


def cal_miou(img1, img2, skip_bg=True):

    classnum = int(img2.max())+1
    iou = np.zeros(classnum)
    for i in range(0, classnum):
        imga = img1 == i
        imgb = img2 == i
        imgi = imga * imgb
        imgu = imga + imgb
        iou[i] = np.sum(imgi) / np.sum(imgu)
    if skip_bg:
        iou = iou[1:]

    miou = np.mean(iou)
    return miou, iou


def cal_miou_orig(img1, img2):
    classnum = img2.max()
    iou = np.zeros((int(classnum), 1))
    for i in range(int(classnum)):
        imga = img1 == i + 1
        imgb = img2 == i + 1
        imgi = imga * imgb
        imgu = imga + imgb
        iou[i] = np.sum(imgi) / np.sum(imgu)
    miou = np.mean(iou)
    return miou

def make_one_hot(input, shape):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    result = torch.zeros(shape).to(device = input.device)
    result.scatter_(1, input, 1)
    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        # print("   ", den.max().item(), predict.max().item(), target.max().item())
        # exit()
        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        # self.ignore_index = ignore_index

    def forward(self, predict, target):
        shape = predict.shape
        target = torch.unsqueeze(target, 1)



        target = make_one_hot(target.long(), shape)


        assert predict.shape == target.shape, "predict & target shape do not match"
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        # print()
        # print(predict.shape, target.shape, torch.sum(self.weight))
        # print()
        for i in range(target.shape[1]):
            # if i != self.ignore_index:
            dice_loss = dice(predict[:, i], target[:, i])

            # print(i, dice_loss.item())
            if self.weight is not None:
                assert self.weight.shape[0] == target.shape[1], "Expect weight shape [{}], get[{}]".format(target.shape[1], self.weight.shape[0])
                dice_loss *= self.weight[i]



            total_loss += dice_loss


        divisor = target.shape[1] if self.weight is None else torch.sum(self.weight)
        # print(divisor)
        return total_loss / divisor


def get_patch_random(data, label, cube_size, patch_size):
    """
    :param data: input data
    :param label: input label
    :param cube_size: size of input data
    :param patch_size: size of patch
    :return: cropped data and label

    Usage: Randomly crop a patch from the input data and label
    """
    patch_pos = []
    for i in range(3):
        patch_pos.append(torch.randint(0, cube_size[i] - patch_size[i] + 1, (1,)))
        # print(cube_size[i], patch_size[i], patch_pos[i])
    data_crop = data[:, :, patch_pos[0] : patch_pos[0] + patch_size[0],
                     patch_pos[1] : patch_pos[1] + patch_size[1], patch_pos[2] : patch_pos[2] + patch_size[2]]
    # print("\t data crop:")
    # print("\t \t Before: min = ", torch.min(data_crop).item(), "max = ", torch.max(data_crop).item())
    # data_crop = data_crop.contiguous()
    # print("\t \t After: min = ", torch.min(data_crop).item(), "max = ", torch.max(data_crop).item())
    label_crop = label[:, :, patch_pos[1] : patch_pos[1] + patch_size[1], patch_pos[2] : patch_pos[2] + patch_size[2]]
    # label_crop = label_crop.contiguous()
    return data_crop, label_crop


def split_test(data, model, cube_size, patch_size, n_classes, ava_classes=None):
    """
    :param data: input data
    :param model: model
    :param cube_size: size of input data
    :param patch_size: size of patch
    :param n_classes: number of classes
    :return: output of model

    Usage: Iteratively split the input data into patches and feed them into the model,
    then combine the outputs of the model to get the final output
    """
    outshape = [1, n_classes, 1, cube_size[1], cube_size[2]]
    result_cavf = torch.zeros(outshape)

    if ava_classes is not None:
        outshape = [1, ava_classes, 1, cube_size[1], cube_size[2]]
        result_ava = torch.zeros(outshape)
    else:
        result_ava = None
    result_cavf = result_cavf.to(data.device)

    i = 0
    for x in range(0, cube_size[0], patch_size[0]):
        for y in range(0, cube_size[1], patch_size[1]):
            for z in range(0, cube_size[2], patch_size[2]):
                i += 1
                input = data[:, :, x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]]
                output_cavf, output_ava = model(input)
                result_cavf[:, :, x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]] = output_cavf

                if output_ava is not None:
                    result_ava[:, :, x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]] = output_ava
    assert i == 16, "The number of patches should be 16"
    return result_cavf, result_ava





def save_slices(imgs, titles, dim, save_path, img_name, num_slices = 5):
    """
        Plot and save slices of images along a given dimension

        Args:
            - imgs: list of 3D images. Allows 2D images as well but no slices will be taken.
                    3D images should be of shape (D, H, W), and 2D images should be of shape (H, W)
            - titles: list of titles for each image
            - dim: dimension along which to take the slices
            - save_path: path to save the images
            - img_name: name of the image to save.
            - num_slices: number of slices to take along the given dimension

        Returns:
            - None, saves all slices will be saved in folder "save_path/img_name"
    """
    assert dim in [0, 1, 2], "The dimension should be 0, 1 or 2"
    assert len(imgs) == len(titles), "The number of images should be equal to the number of titles"
    # assert imgs[0].ndim == 3, "The input image should be 3D"
    D, H, W = imgs[0].shape

    n = len(imgs)

    save_folder = os.path.join(save_path, img_name)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    if dim == 0:
        for i in range(0, D, D//num_slices):

            fig, axes = plt.subplots(1, n, figsize=(n*5, 5))

            for idx, (img, title) in enumerate(zip(imgs, titles)):

                if img.ndim == 3:
                    im1 = axes[idx].imshow(img[i, :, :], cmap='gray')
                elif img.ndim == 2:
                    im1 = axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(title)
                # fig.colorbar(im1, ax=axes[idx])

            fig.colorbar(im1, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
            fig.suptitle(f"{img_name} Slice D = {i}")
            fig.savefig(save_folder + f"/D_slice_{i}.png")
            plt.close('all')

    elif dim == 1:
        for i in range(0, H, H//num_slices):
            fig, axes = plt.subplots(1, n, figsize=(n*5, 5))

            for idx, (img, title) in enumerate(zip(imgs, titles)):

                if img.ndim == 3:
                    im1 = axes[idx].imshow(img[:, i, :], cmap='gray')
                elif img.ndim == 2:
                    im1 = axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(title)
                # fig.colorbar(im1, ax=axes[idx])

            fig.colorbar(im1, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
            fig.suptitle(f"{img_name} Slice H = {i}")
            fig.savefig(save_folder + f"/H_slice_{i}.png")
            plt.close('all')

    else:
        for i in range(0, W, W//num_slices):
            fig, axes = plt.subplots(1, n, figsize=(n*5, 5))

            for idx, (img, title) in enumerate(zip(imgs, titles)):

                if img.ndim == 3:
                    im1 = axes[idx].imshow(img[:, :, i], cmap='gray')
                elif img.ndim == 2:
                    im1 = axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(title)


                # fig.colorbar(im1, ax=axes[idx])

            fig.colorbar(im1, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
            fig.suptitle(f"{img_name} Slice W = {i}")
            fig.savefig(save_folder + f"/W_slice_{i}.png")
            plt.close('all')

    plt.close('all')



def save_histogram(img, save_path, title, bins = 30):
    """
        Plot and save histogram of an image

        Args:
            - img: any image.
            - save_path: path to save the images
            - title: title of the histogram
            - bins: number of bins for the histogram
    """

    plt.figure()
    plt.hist(img.flatten(), bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.savefig(save_path)
    plt.close()



def get_ava_RGB(regions, overlay_image = None):
    assert regions.ndim == 2
    assert overlay_image is None or overlay_image.ndim == 2
    # Initialize an empty BGR image

    RGB_img = np.zeros((regions.shape[0], regions.shape[1], 3), dtype=np.uint8)


    artery_regions = np.argwhere(regions == 0).T
    vein_regions = np.argwhere(regions == 1).T

    RGB_img[artery_regions[0], artery_regions[1], :] = [200, 120, 120]
    RGB_img[vein_regions[0], vein_regions[1], :] = [120, 120, 200]


    if overlay_image is not None:
        overlay_artery_coords = np.argwhere(overlay_image == 2).T
        overlay_vein_coords = np.argwhere(overlay_image == 3).T
        overlay_faz_coords = np.argwhere(overlay_image == 4).T
        RGB_img[overlay_artery_coords[0], overlay_artery_coords[1], :] = [255, 0, 0]
        RGB_img[overlay_vein_coords[0], overlay_vein_coords[1], :] = [0, 0, 255]
        RGB_img[overlay_faz_coords[0], overlay_faz_coords[1], :] = [0, 255, 0]

    return RGB_img


def save_ava_image(regions, save_path, name, overlay_image = None):
    """
        Save a BGR image of the regions

        Args:
            - regions: 2D numpy array with 0 representing artery region and 1 representing vein region
            - save_path: path to save the image
            - name: name of the image
            - overlay_image: 2D numpy array in which 2 represents artery, 3 represents vein, and 4 represents FAZ

        Returns:
            - None, saves the image of regions at "save_path/name.bmp" with the artery, veins, and FAZ overlayed if given

    """
    # Clip values to 0-255 and convert to uint8
    RGB_img = get_ava_RGB(regions, overlay_image)

    # Save the image
    Image.fromarray(RGB_img).save(os.path.join(save_path, f"{name}.bmp"))
    # cv2.imwrite(os.path.join(save_path, f"{name}.bmp"), BGR_img)


def get_cavf_RGB(image):
    assert image.ndim == 4 or image.ndim == 3
    if image.ndim == 4:
        image = np.squeeze(image, axis=0)


    RGB_img = np.zeros((image.shape[1], image.shape[2], 3))

    RGB_img[:, :, 0] = 15 * image[0, :, :] + 171 * image[1, :, :] + 215 * image[2, :, :] + 43 * image[3, :, :] + 166 * image[4, :, :]
    RGB_img[:, :, 1] = 32 * image[0, :, :] + 165 * image[1, :, :] + 25 * image[2, :, :] + 131 * image[3, :, :] + 217 * image[4, :, :]
    RGB_img[:, :, 2] = 53 * image[0, :, :] + 143 * image[1, :, :] + 28 * image[2, :, :] + 186 * image[3, :, :] + 106 * image[4, :, :]

    return RGB_img.astype(np.uint8)

def save_cavf_image(image, save_path, name):
    """
        Save a BGR image of the CAVF regions

        Args:
            - image: 3D or 4D numpy array of shape (C, H, W) or (1, C, H, W) with C representing the number of classes. C should be 5.
            - save_path: path to save the image
            - name: name of the image

        Returns:
            - None, saves the image of regions at "save_path/name.bmp"
    """

    RGB_img = get_cavf_RGB(image)
    Image.fromarray(RGB_img).save(os.path.join(save_path, f"{name}.bmp"))
    # cv2.imwrite(os.path.join(save_path, f"{name}.bmp"), BGR)




def normalize(img):
    if type(img) == torch.Tensor:
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    elif type(img) == np.ndarray:
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        raise ValueError("Input image must be a numpy array or a torch tensor.")

def one_hot_encode(argmax, num_classes):
    """must be shape (C, H, W)"""
    return (np.eye(num_classes)[argmax]).transpose(2, 0, 1)





def parse_old_IPNV2_state_dict(path):

    state_dict = torch.load(path)

    gn1_keys = [k for k in state_dict.keys() if "gn1" in k]
    for key in gn1_keys:
        new_key = key.replace("gn1", "norm1")
        state_dict[new_key] = state_dict.pop(key)

    gn2_keys = [k for k in state_dict.keys() if "gn2" in k]
    for key in gn2_keys:
        new_key = key.replace("gn2", "norm2")
        state_dict[new_key] = state_dict.pop(key)

    return state_dict


def parse_old_GAN_state_dict(path):
    model_states = torch.load(path)
    for mdl in model_states.keys():
        model_dict = model_states[mdl]
        gn1_keys = [k for k in model_dict.keys() if "gn1" in k]

        for key in gn1_keys:
            new_key = key.replace("gn1", "norm1")
            model_dict[new_key] = model_dict.pop(key)

        gn2_keys = [k for k in model_dict.keys() if "gn2" in k]
        for key in gn2_keys:
            new_key = key.replace("gn2", "norm2")
            model_dict[new_key] = model_dict.pop(key)

        if mdl == "net_G":
            SegNet2D_keys = [k for k in model_dict.keys() if "SegNet2D" in k]
            for key in SegNet2D_keys:
                model_dict.pop(key)
    return model_states
