import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

class AireadiMaskDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.subfolders = []

        for folder in os.listdir(root):
            full_folder = osp.join(root, folder)
            if osp.isdir(full_folder) and folder.startswith("patient_"):
                self.subfolders.append(full_folder)

        self.subfolders.sort()

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        folder = self.subfolders[index]
        basename = os.path.basename(folder)
        parts = basename.split('_')
        if len(parts) < 5:
            raise ValueError(f"Folder name {basename} does not match expected format 'patient_{{patient_id}}_{{L or R}}_anchor_{{i}}'.")
        patient_id = parts[1]
        side = parts[2]
        anchor = parts[4]

        subitems = os.listdir(folder)
        masks = []
        if "fail" in subitems and osp.isdir(osp.join(folder, "fail")):
            fail_dir = osp.join(folder, "fail")
            success_dir = osp.join(folder, "success")
            fail_files = sorted([osp.join(fail_dir, f) for f in os.listdir(fail_dir) if f.endswith('.npy')])
            success_files = sorted([osp.join(success_dir, f) for f in os.listdir(success_dir) if f.endswith('.npy')])
            for fp in fail_files:
                masks.append(np.load(fp))
            for fp in success_files:
                masks.append(np.load(fp))
        else:
            npy_files = sorted([osp.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')])
            for fp in npy_files:
                masks.append(np.load(fp))

        sample = {
            "patient_id": patient_id,
            "side": side,
            "anchor": anchor,
            "masks": masks,
            "folder": folder
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
