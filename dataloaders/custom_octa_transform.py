import torch
import random
import torch.nn.functional as F
from .utils import normalize

class Custom3DTransformTrain(object):
    def __init__(self, resize_size=256, noise_prob=0.5, gaussian_std=0.05, light_adjust=0.2):
        self.resize_size = resize_size
        self.noise_prob = noise_prob
        self.gaussian_std = gaussian_std
        self.light_adjust = light_adjust

    def __call__(self, sample):
        image = sample['image'].squeeze(1)
        label = sample['proj_map'].squeeze(1)

        image = F.interpolate(image, size=(image.shape[2], self.resize_size, self.resize_size),
                              mode='trilinear', align_corners=True)

        if label is not None:
            if label.ndim == 4:
                label = F.interpolate(label, size=(self.resize_size, self.resize_size), mode='nearest')
            else:
                raise ValueError("Unexpected label dimensions: {}".format(label.ndim))


        image_2d = image[:, 0:1, :, :, :]
        image_3d = image[:, 1:2, :, :, :]

        two_d_was_squeezed = False
        if image_2d.shape[2] == 1:
            image_2d = image_2d.squeeze(2)
            two_d_was_squeezed = True

        if random.random() < 0.5:
            self.gaussian_std = 0.1

        noise_2d = torch.randn_like(image_2d) * self.gaussian_std
        image_2d = image_2d + noise_2d

        noise_3d = torch.randn_like(image_3d) * self.gaussian_std
        image_3d = image_3d + noise_3d

        contrast_factor = 1.0 + (random.random() - 0.5) * 2 * self.light_adjust
        image_2d = image_2d * contrast_factor

        if two_d_was_squeezed:
            image_2d = image_2d.unsqueeze(2)

        image = torch.cat([image_2d, image_3d], dim=1)

        for c in range(image.shape[1]):
            image[:, c, :, :, :] = normalize(image[:, c, :, :, :])

        sample['image'] = image
        sample['proj_map'] = label
        return sample


class Custom3DTransformWeak(object):

    def __init__(self, resize_size=256):
        self.resize_size = resize_size

    def __call__(self, sample):
        image = sample['image']
        proj_map = sample['proj_map']
        label = sample['data_label']

        image = F.interpolate(
            image,
            size=(image.shape[2], self.resize_size, self.resize_size),
            mode='trilinear',
            align_corners=True
        )

        if proj_map is not None:
            if proj_map.ndim == 4:
                proj_map = F.interpolate(
                    proj_map,
                    size=(self.resize_size, self.resize_size),
                    mode='nearest'
                )
            else:
                raise ValueError("Unexpected label dimensions: {}".format(proj_map.ndim))

        if image.ndim == 4:
            image = image.unsqueeze(0)

        for c in range(image.shape[1]):
            image[:, c, :, :, :] = normalize(image[:, c, :, :, :])

        sample['image'] = image
        sample['proj_map'] = proj_map

        try:
            if label is not None and not torch.is_tensor(label):
                label = torch.from_numpy(label)

            if label.ndim == 2:
                label = label.unsqueeze(0).unsqueeze(0)

            label = label.float()
            label = F.interpolate(
                label,
                size=(self.resize_size, self.resize_size),
                mode='nearest'
            )
            label = label.long()
            label = label.squeeze(0)
            sample['data_label'] = label
        except Exception as e:
            return sample

        return sample