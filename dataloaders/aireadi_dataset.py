import pydicom
import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from .utils import normalize
from skimage.filters import threshold_otsu
import torch.nn.functional as F
import csv


def load_roi_cache(csv_file):
    """
    Load ROI cache from a CSV file.
    Returns a dictionary mapping image_hash to a tuple (dmin, dmax, roi_depth).
    If the CSV file doesn't exist, returns an empty dictionary.
    """
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, dtype={'dmin': int, 'dmax': int, 'roi_depth': int})
        df = df.set_index('idx')
        cache = df.to_dict('index')
        return cache
    else:
        return {}

def save_roi_to_csv(csv_file, idx, dmin, dmax, roi_depth):
    """
    Append a new ROI entry into the CSV file.
    """
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ['idx', 'dmin', 'dmax', 'roi_depth']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'idx': idx,
            'dmin': dmin,
            'dmax': dmax,
            'roi_depth': roi_depth
        })


def get_ROI(img, idx, target_depth=550,  cache_file="roi_cache.csv"):
    """
        Get the region of interest (ROI) of the input image

        Args:
            - img: 5D tensor of shape (1, 2, D, H, W). where the first channel is OCT and the second channel is OCTA
            - target_depth: target depth of the ROI.
                            Assumes target_depth will be larger than calculated otsu threshold roi depth.

            - Returns:
                - 5D tensor of shape (1, 2, roi_depth, H, W). Note that roi_depth may not be equal to target_depth
                    if target depth is smaller than the calculated roi depth.
    """
    assert img.ndim == 5, "Input image must be 5D = (B, C, D, H, W)."
    assert img.shape[1] == 2, "Input image must have 2 channels, with the first channel being OCT and the second channel being OCTA."


    cache = load_roi_cache(cache_file)
    if idx in cache:
        dmin, dmax, roi_depth = cache[idx]['dmin'], cache[idx]['dmax'], cache[idx]['roi_depth']
        # print(f"Using cached ROI: dmin={dmin}, dmax={dmax}, roi_depth={roi_depth}")
        return img[:, :, dmin:dmax + 1, :, :]
    print(f"Calculating ROI for {idx}...")


    OCT_img = img[0, 0, :, :, :]
    OCTA_img = img[0, 1, :, :, :]


    thresh_OCT = threshold_otsu(OCT_img.cpu().numpy())
    thresh_OCTA = threshold_otsu(OCTA_img.cpu().numpy())

    mask_OCT = OCT_img > thresh_OCT
    mask_OCTA = OCTA_img > thresh_OCTA
    dmin, dmax = 0, img.shape[2]

    low_thresh = 0
    high_thresh = torch.numel(mask_OCT)

    while dmax - dmin > target_depth:
        # finding depths with more than sum_thresh pixels for OCT and OCTA image
        sum_thresh = low_thresh + (high_thresh - low_thresh) // 2
        # print(sum_thresh)

        # print(len(torch.where(torch.sum(mask_OCT, axis=(1, 2)) > sum_thresh)[0]))
        depths_OCT = torch.where(torch.sum(mask_OCT, axis=(1, 2)) > sum_thresh)[0]
        depths_OCTA = torch.where(torch.sum(mask_OCTA, axis=(1, 2)) > sum_thresh)[0]

        if depths_OCT.shape[0] == 0 or depths_OCTA.shape[0] == 0:
            high_thresh = sum_thresh - 1
            continue
        # dmin_OCT, dmax_OCT = torch.where(torch.sum(mask_OCT, axis=(1, 2)) > sum_thresh)[0][[0, -1]]
        # dmin_OCTA, dmax_OCTA = torch.where(torch.sum(mask_OCTA, axis=(1, 2)) > sum_thresh)[0][[0, -1]]

        dmin_OCT, dmax_OCT = depths_OCT[[0, -1]]
        dmin_OCTA, dmax_OCTA = depths_OCTA[[0, -1]]

        # take largest range of depths
        dmin = max(dmin_OCT.item(), dmin_OCTA.item())
        dmax = min(dmax_OCT.item(), dmax_OCTA.item())

        if dmax - dmin > target_depth:
            low_thresh = sum_thresh + 1

        # increase the threshold and run again if the depth is still larger than target depth
        # sum_thresh += 1000

    # expand the depth range to target depth if the depth is smaller than target depth
    roi_depth = dmax - dmin + 1
    # print(dmin, dmax, roi_depth)
    if roi_depth <= target_depth:
        offset = (target_depth - roi_depth) // 2
        dmin = max(0, dmin - offset)
        dmax = min(img.shape[2] - 1, dmax + offset)

        current_depth = dmax - dmin + 1
        if  current_depth < target_depth:
            dmax = min(img.shape[2] - 1, dmax + (target_depth - current_depth))

    save_roi_to_csv(cache_file, idx, dmin, dmax, dmax - dmin + 1)

    return img[:, :, dmin:dmax + 1, :, :]


def process_data(OCT_img, OCTA_img, device, idx, input_shape=(128, 256, 256),
                 roi_target_depth=550, use_proj_map = False,
                 OCT_proj_map = None, OCTA_proj_map = None):

    OCT_img = torch.from_numpy(OCT_img).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    OCTA_img = torch.from_numpy(OCTA_img).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

    # concatenate the OCT and OCTA images along the channel dimension
    data = torch.cat((OCT_img, OCTA_img), dim=1)

    # get region of interest by cropping depth to target depth
    data = get_ROI(data, idx, roi_target_depth)

    # normalize the data along channels
    for i in range(data.shape[1]):
        data[0, i, :, :, :] = normalize(data[0, i, :, :, :])

    # resize the data to the input shape
    data = F.interpolate(data, size=input_shape, mode='trilinear', align_corners=True)

    # normalize the data after interpolation along channels.
    for i in range(data.shape[1]):
        data[0, i, :, :, :] = normalize(data[0, i, :, :, :])

    H, W = input_shape[1], input_shape[2]

    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    manhattan_map = (np.abs(u_grid - 0.5 * W) + np.abs(v_grid - 0.5 * H)) / (0.5 * (H + W))

    manhattan_map_tensor = torch.from_numpy(manhattan_map).to(data.device).float()

    data[0, :, 0, :, :] = manhattan_map_tensor

    proj_map = None
    if use_proj_map:
        # generate the projection map using mean along depth if projection map is not provided
        if OCTA_proj_map is None:
            raise ValueError("OCTA projection map must be provided if use_proj_map is True")
        else:
            OCTA_proj_map = torch.from_numpy(OCTA_proj_map).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

        if OCT_proj_map is None:
            OCT_proj_map = torch.mean(data[0, 0, :, :, :], dim=0).unsqueeze(0).unsqueeze(0)
        else:
            OCT_proj_map = torch.from_numpy(OCT_proj_map).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

        # normalize the projection maps
        OCTA_proj_map = normalize(OCTA_proj_map)
        OCT_proj_map = normalize(OCT_proj_map)

        OCT_proj_map = F.interpolate(OCT_proj_map, size=(input_shape[1], input_shape[2]), mode='bilinear', align_corners=True)
        OCTA_proj_map = F.interpolate(OCTA_proj_map, size=(input_shape[1], input_shape[2]), mode='bilinear', align_corners=True)

        OCTA_proj_map = normalize(OCTA_proj_map)
        OCT_proj_map = normalize(OCT_proj_map)

        # concatenate the projection maps along the channel dimension
        proj_map = torch.cat((OCT_proj_map, OCTA_proj_map), dim=1)

    return data, proj_map


class aireadi_dataset:
    def __init__(self, root, roi, device, mode='train', all_success=False, fail_image_path=None, npz_path=None):
        super().__init__()
        self.root = root
        self.roi = roi
        self.device = device
        self.mode = mode
        self.octa_dir = root
        self.test_manifest_path = None

        if mode == 'train' and all_success:
            octa_manifest_tsv = self.octa_dir + 'success_manifest_train.tsv'
        elif mode == 'test' and all_success:
            octa_manifest_tsv = self.octa_dir + 'success_manifest_test.tsv'
            self.test_set_dir = self.octa_dir + 'mini_test_set/'
            test_manifest_path = self.test_set_dir + 'manifest.tsv'
        elif mode == 'train':
            octa_manifest_tsv = self.octa_dir + 'manifest_train.tsv'
        elif mode == 'test':
            octa_manifest_tsv = self.octa_dir + 'success_manifest_test.tsv'
            self.test_set_dir = self.octa_dir + 'mini_test_set/'
            test_manifest_path = self.test_set_dir + 'manifest.tsv'
        else:
            octa_manifest_tsv = self.octa_dir + 'manifest.tsv'


        full_manifest = pd.read_csv(octa_manifest_tsv, sep='\t')

        split_manifest = full_manifest

        if mode == 'train':

            if not fail_image_path or os.path.exists(fail_image_path):

                fail_df = pd.read_csv(fail_image_path, dtype=str, sep='\t')

                for col in ['patient_id', 'manufacturer_model_name', 'laterality', 'anatomic_region']:
                    fail_df[col] = fail_df[col].str.strip()

                for col in ['participant_id', 'manufacturers_model_name', 'laterality', 'anatomic_region']:
                    if col not in full_manifest.columns:
                        raise ValueError(f"The manifest must contain a '{col}' column for splitting and filtering.")
                    split_manifest[col] = split_manifest[col].astype(str).str.strip()

                split_manifest = split_manifest.copy()
                split_manifest['key'] = (
                    split_manifest['participant_id'] + '_' +
                    split_manifest['manufacturers_model_name'] + '_' +
                    split_manifest['laterality'] + '_' +
                    split_manifest['anatomic_region']
                )
                fail_df['key'] = (
                    fail_df['patient_id'] + '_' +
                    fail_df['manufacturer_model_name'] + '_' +
                    fail_df['laterality'] + '_' +
                    fail_df['anatomic_region']
                )

                before_count = len(split_manifest)
                split_manifest = split_manifest[~split_manifest['key'].isin(fail_df['key'])].reset_index(drop=True)
                after_count = len(split_manifest)
                print(f"Filtered out {before_count - after_count} rows from manifest due to failed images (after splitting).")
        else:
            self.test_manifest = pd.read_csv(test_manifest_path, sep='\t')

            for col in ['participant_id', 'manufacturer_model_name', 'laterality', 'anatomic_region']:
                self.test_manifest[col] = self.test_manifest[col].astype(str).str.strip()

            for col in ['participant_id', 'manufacturers_model_name', 'laterality', 'anatomic_region']:
                if col not in full_manifest.columns:
                    raise ValueError(f"The manifest must contain a '{col}' column for splitting and filtering.")
                split_manifest[col] = split_manifest[col].astype(str).str.strip()

            self.test_manifest['key'] = (
                self.test_manifest['participant_id'] + '_' +
                self.test_manifest['manufacturer_model_name'] + '_' +
                self.test_manifest['laterality'] + '_' +
                self.test_manifest['anatomic_region']
            )

            split_manifest['key'] = (
                split_manifest['participant_id'] + '_' +
                split_manifest['manufacturers_model_name'] + '_' +
                split_manifest['laterality'] + '_' +
                split_manifest['anatomic_region']
            )

            split_manifest = split_manifest[split_manifest['key'].isin(self.test_manifest['key'])].reset_index(drop=True)
            self.test_manifest.drop(columns=['key'], inplace=True)

        self.manifest = split_manifest.copy()
        self.manifest.drop(columns=['key'], inplace=True)

        if os.path.exists(npz_path):
            pseudo_npz = np.load(npz_path, allow_pickle=True)
            self.proto_pseudo_dic = pseudo_npz['arr_2'].item()
        else:
            self.proto_pseudo_dic = {}


    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if idx >= len(self.manifest):
            raise IndexError

        row = self.manifest.iloc[idx]

        oct_fp = self.root + row['associated_structural_oct_file_path']
        octa_fp = self.root + row['flow_cube_file_path']
        enface_fp = self.root + row['associated_enface_1_file_path']


        OCT_img = pydicom.dcmread(oct_fp).pixel_array
        OCTA_img = pydicom.dcmread(octa_fp).pixel_array

        OCT_img = OCT_img.transpose(1, 0, 2)
        OCTA_img = OCTA_img.transpose(1, 0, 2)

        enface_img = pydicom.dcmread(enface_fp).pixel_array

        data, proj_map = process_data(OCT_img=OCT_img, OCTA_img=OCTA_img, device=self.device, idx=row['associated_structural_oct_file_path'],
                     roi_target_depth=self.roi, use_proj_map=True, OCTA_proj_map=enface_img)

        manufacturer_map = {"Maestro2": 0, "Triton": 1, "Cirrus": 2}
        anatomical_map = {"Macula": 0, "Optic Disc": 1}
        region_size_map = {"6 x 6": 0, "12 x 12": 1}
        laterality_map = {"L": 0, "R": 1}

        manufacturer_label = manufacturer_map[row["manufacturers_model_name"]]

        region_str = row["anatomic_region"]
        parts = [s.strip() for s in region_str.split(",")]
        anatomical_label = anatomical_map[parts[0]]
        region_size_label = region_size_map[parts[1]]

        laterality_label = laterality_map[row["laterality"]]

        if self.mode == 'test':
            #TODO: change the path to your test set label
            data_label = np.load(self.test_set_dir + f"path/to/label.npy")
        else:
            data_label = 1

        filename = row['associated_enface_1_file_path']
        base = os.path.basename(filename)
        filename_without_ext, _ = os.path.splitext(base)

        if self.proto_pseudo_dic is not None:
            proto_pseudo_npz = self.proto_pseudo_dic.get(filename_without_ext, '')
        else:
            proto_pseudo_npz = ''

        #TODO: change the path to your intergrated softmax label for individual image
        softmax_dir = 'path to yout intergrated softmax label for individual image'

        softmax_path = os.path.join(softmax_dir, "softmax.npy")
        if os.path.exists(softmax_path):
            merge_softmax_np = np.load(softmax_path)
        else:
            raise FileNotFoundError(f"Label file not found: {softmax_path}")

        return data, proj_map, row, manufacturer_label, anatomical_label, region_size_label, laterality_label, data_label, merge_softmax_np, proto_pseudo_npz


class AireadiParticipantDataset(Dataset):
    """
    This dataset groups all entries (rows) from the manifest that share the same
    participant_id. For each participant, __getitem__ returns a dictionary containing:
      - 'participant_id': the participant id (taken from the manifest), and
      - 'samples': a list of dictionaries, one for each imaging session for that participant.
        Each sample dictionary has keys such as 'image', 'proj_map', and several labels.
    """
    def __init__(self, root, roi, device, mode='train', all_success=False, fail_image_path=None, npz_path=None):
        super().__init__()
        self.root = root
        self.roi = roi
        self.device = device
        self.mode = mode
        self.octa_dir = root
        self.test_manifest_path = None

        if mode == 'train' and all_success:
            octa_manifest_tsv = self.octa_dir + 'success_manifest_train.tsv'
        elif mode == 'test' and all_success:
            octa_manifest_tsv = self.octa_dir + 'success_manifest_test.tsv'
            self.test_set_dir = self.octa_dir + 'mini_test_set/'
            test_manifest_path = self.test_set_dir + 'manifest.tsv'
        elif mode == 'train':
            octa_manifest_tsv = self.octa_dir + 'manifest_train.tsv'
        elif mode == 'test':
            octa_manifest_tsv = self.octa_dir + 'success_manifest_test.tsv'
            self.test_set_dir = self.octa_dir + 'mini_test_set/'
            test_manifest_path = self.test_set_dir + 'manifest.tsv'
        else:
            octa_manifest_tsv = self.octa_dir + 'manifest.tsv'


        full_manifest = pd.read_csv(octa_manifest_tsv, sep='\t')

        split_manifest = full_manifest

        if mode == 'train':

            if not fail_image_path or os.path.exists(fail_image_path):

                fail_df = pd.read_csv(fail_image_path, dtype=str, sep='\t')

                for col in ['patient_id', 'manufacturer_model_name', 'laterality', 'anatomic_region']:
                    fail_df[col] = fail_df[col].str.strip()

                for col in ['participant_id', 'manufacturers_model_name', 'laterality', 'anatomic_region']:
                    if col not in full_manifest.columns:
                        raise ValueError(f"The manifest must contain a '{col}' column for splitting and filtering.")
                    split_manifest[col] = split_manifest[col].astype(str).str.strip()

                split_manifest = split_manifest.copy()
                split_manifest['key'] = (
                    split_manifest['participant_id'] + '_' +
                    split_manifest['manufacturers_model_name'] + '_' +
                    split_manifest['laterality'] + '_' +
                    split_manifest['anatomic_region']
                )
                fail_df['key'] = (
                    fail_df['patient_id'] + '_' +
                    fail_df['manufacturer_model_name'] + '_' +
                    fail_df['laterality'] + '_' +
                    fail_df['anatomic_region']
                )

                before_count = len(split_manifest)
                split_manifest = split_manifest[~split_manifest['key'].isin(fail_df['key'])].reset_index(drop=True)
                after_count = len(split_manifest)
                print(f"Filtered out {before_count - after_count} rows from manifest due to failed images (after splitting).")
        else:
            self.test_manifest = pd.read_csv(test_manifest_path, sep='\t')

            for col in ['participant_id', 'manufacturer_model_name', 'laterality', 'anatomic_region']:
                self.test_manifest[col] = self.test_manifest[col].astype(str).str.strip()

            for col in ['participant_id', 'manufacturers_model_name', 'laterality', 'anatomic_region']:
                if col not in full_manifest.columns:
                    raise ValueError(f"The manifest must contain a '{col}' column for splitting and filtering.")
                split_manifest[col] = split_manifest[col].astype(str).str.strip()

            self.test_manifest['key'] = (
                self.test_manifest['participant_id'] + '_' +
                self.test_manifest['manufacturer_model_name'] + '_' +
                self.test_manifest['laterality'] + '_' +
                self.test_manifest['anatomic_region']
            )

            split_manifest['key'] = (
                split_manifest['participant_id'] + '_' +
                split_manifest['manufacturers_model_name'] + '_' +
                split_manifest['laterality'] + '_' +
                split_manifest['anatomic_region']
            )

            split_manifest = split_manifest[split_manifest['key'].isin(self.test_manifest['key'])].reset_index(drop=True)
            self.test_manifest.drop(columns=['key'], inplace=True)

        self.manifest = split_manifest.copy()
        self.manifest.drop(columns=['key'], inplace=True)

        if os.path.exists(npz_path):
            pseudo_npz = np.load(npz_path, allow_pickle=True)
            self.proto_pseudo_dic = pseudo_npz['arr_2'].item()
        else:
            self.proto_pseudo_dic = {}

        self.grouped = self.manifest.groupby('participant_id')
        self.participant_ids = list(self.grouped.groups.keys())


    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idx):
        participant_id = self.participant_ids[idx]
        group_df = self.grouped.get_group(participant_id)
        samples = []

        for _, row in group_df.iterrows():

            oct_fp   = self.root + row['associated_structural_oct_file_path']
            octa_fp  = self.root + row['flow_cube_file_path']
            enface_fp = self.root + row['associated_enface_1_file_path']

            OCT_img = pydicom.dcmread(oct_fp).pixel_array
            OCTA_img = pydicom.dcmread(octa_fp).pixel_array
            OCT_img  = OCT_img.transpose(1, 0, 2)
            OCTA_img = OCTA_img.transpose(1, 0, 2)
            enface_img = pydicom.dcmread(enface_fp).pixel_array

            data, proj_map = process_data(
                OCT_img=OCT_img,
                OCTA_img=OCTA_img,
                device=self.device,
                idx=row['associated_structural_oct_file_path'],
                roi_target_depth=self.roi,
                use_proj_map=True,
                OCTA_proj_map=enface_img
            )

            manufacturer_map = {"Maestro2": 0, "Triton": 1, "Cirrus": 2}
            anatomical_map = {"Macula": 0, "Optic Disc": 1}
            region_size_map = {"6 x 6": 0, "12 x 12": 1}
            laterality_map = {"L": 0, "R": 1}

            manufacturer_label = manufacturer_map.get(row["manufacturers_model_name"], -1)
            region_str = row["anatomic_region"]
            parts = [s.strip() for s in region_str.split(",")]
            anatomical_label = anatomical_map.get(parts[0], -1)
            region_size_label = region_size_map.get(parts[1], -1) if len(parts) > 1 else -1
            laterality_label = laterality_map.get(row["laterality"], -1)

            if self.mode == 'test':
                # TODO: change the path to your test set label
                data_label = np.load(self.test_set_dir + f"/path/to/label.npy")
            else:
                data_label = 1

            filename = row['associated_enface_1_file_path']
            base = os.path.basename(filename)
            filename_without_ext, _ = os.path.splitext(base)

            #TODO: change the path to your intergrated softmax label for individual image
            softmax_dir = 'path to yout intergrated softmax label for individual image'
            softmax_path = os.path.join(softmax_dir, "softmax.npy")
            if os.path.exists(softmax_path):
                merge_softmax_np = np.load(softmax_path)
            else:
                merge_softmax_np = None
                raise FileNotFoundError(f"Label file not found: {softmax_path}")

            sample = {
                'image': data,
                'proj_map': proj_map,
                'img_name': [filename_without_ext],
                'manufacturer': manufacturer_label,
                'anatomical': anatomical_label,
                'laterality': laterality_label,
                'region_size': region_size_label,
                'data_label': data_label,
                'participant_id': row['participant_id'],
                'row': row,
                'merge_softmax_label': merge_softmax_np,
            }
            samples.append(sample)

        return participant_id, samples
