import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from .aireadi_dataset import aireadi_dataset, AireadiParticipantDataset
from torch.utils.data.dataloader import default_collate
import os

class ResumeSampler(Sampler):
    """
    A sampler that supports both:
    1. Resuming from a specific batch index (`resume_idx`).
    2. Deterministic shuffling across epochs (`set_epoch`).
    """
    def __init__(self, data_source, resume_idx=0, generator=None):
        self.data_source = data_source
        self.resume_idx = resume_idx  # Where to resume
        self.generator = generator if generator is not None else torch.Generator()
        self.epoch = 0  # Default epoch

    def set_epoch(self, epoch):
        """
        Set the epoch to control deterministic shuffling.
        This should be called before each epoch.
        """
        self.epoch = epoch
        self.generator.manual_seed(epoch)  # Reset shuffle seed for consistency

    def set_resume_idx(self, idx):
        """
        Update the resume index to skip processed data.
        """
        self.resume_idx = idx

    def __iter__(self):
        """
        Generate indices:
        - Shuffled deterministically per epoch.
        - Starts from `resume_idx` to avoid reloading previous data.
        """
        indices = list(range(len(self.data_source)))  # Full dataset indices
        torch.manual_seed(self.epoch)  # Ensure deterministic shuffling
        indices = torch.randperm(len(indices), generator=self.generator).tolist()  # Shuffle

        indices = indices[self.resume_idx:]  # Skip already processed data
        return iter(indices)

    def __len__(self):
        return len(self.data_source) - self.resume_idx  # Adjusted length


class AireadiSegmentation(Dataset):
    def __init__(self, root, roi, device, mode='train', transform=None, all_success=False, fail_image_path=None, npz_path=None):
        self.dataset = aireadi_dataset(root, roi, device, mode, all_success=all_success, fail_image_path=fail_image_path, npz_path=npz_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, proj_map, row, manufacturer, anatomical, region_size, laterality, data_label, merge_softmax_label, proto_pseudo_npz = self.dataset[idx]

        filename = row['associated_enface_1_file_path']
        base = os.path.basename(filename)
        filename_without_ext, _ = os.path.splitext(base)

        sample = {
            'image': data,
            'proj_map': proj_map,
            'img_name': filename_without_ext,
            'manufacturer': manufacturer,
            'anatomical': anatomical,
            'laterality': laterality,
            'region_size': region_size,
            'data_label': data_label,
            'participant_id': row['participant_id'],
            'merge_softmax_label': merge_softmax_label,
            'row': row.to_dict(),
            'proto_pseudo_npz': proto_pseudo_npz,
        }
        if self.transform is not None:
            sample = self.transform(sample)

        if sample is None:
            raise ValueError(f"Sample at index {idx} is None")
        return sample


class AireadiSegmentation_2transform(Dataset):
    def __init__(self, root, roi, device, mode='train', transform_weak=None, transform_strong=None, all_success=False, fail_image_path=None, npz_path=None):
        self.dataset = aireadi_dataset(root, roi, device, mode, all_success=all_success, fail_image_path=fail_image_path, npz_path=npz_path)
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        data, proj_map, row, manufacturer, anatomical, region_size, laterality, data_label, merge_softmax_label, proto_pseudo_npz = self.dataset[idx]

        filename = row['associated_enface_1_file_path']
        base = os.path.basename(filename)
        filename_without_ext, _ = os.path.splitext(base)

        base_sample = {
            'image': data,
            'proj_map': proj_map,
            'img_name': filename_without_ext,
            'manufacturer': manufacturer,
            'anatomical': anatomical,
            'laterality': laterality,
            'region_size': region_size,
            'data_label': data_label,
            'participant_id': row['participant_id'],
            'merge_softmax_label': merge_softmax_label,
            'row': row.to_dict(),
            'proto_pseudo_npz': proto_pseudo_npz,
        }

        if self.transform_weak is not None:
            sample_weak = self.transform_weak(base_sample)

        if self.transform_strong is not None:
            sample_strong = self.transform_strong(base_sample)

        return sample_weak, sample_strong


class AireadiParticipantSegmentation(Dataset):
    def __init__(self, root, roi, device, mode='train', transform=None, all_success=False, fail_image_path=None, npz_path=None):
        self.dataset = AireadiParticipantDataset(root, roi, device, mode, all_success=all_success, fail_image_path=fail_image_path, npz_path=npz_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        participant_id, samples = self.dataset[idx]
        return_samples = []
        if self.transform is not None:
            for sample in samples:
                sample = self.transform(sample)
                return_samples.append(sample)

        return {
            'participant_id': participant_id,
            'samples': return_samples
        }