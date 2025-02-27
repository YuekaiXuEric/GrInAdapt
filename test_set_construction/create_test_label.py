import os
import numpy as np
import argparse
import pandas as pd
from PIL import Image
import pydicom
import cv2


def to_one_hot(argmax, num_classes):
    return np.eye(num_classes)[argmax]

def get_cavf_RGB(image):
    if np.argmin(image.shape) == 2:
        image = image.transpose(2, 0, 1)

    RGB_img = np.zeros((image.shape[1], image.shape[2], 3))

    RGB_img[:, :, 0] = 15 * image[0, :, :] + 171 * image[1, :, :] + 215 * image[2, :, :] + 43 * image[3, :, :] + 166 * image[4, :, :]
    RGB_img[:, :, 1] = 32 * image[0, :, :] + 165 * image[1, :, :] + 25 * image[2, :, :] + 131 * image[3, :, :] + 217 * image[4, :, :]
    RGB_img[:, :, 2] = 53 * image[0, :, :] + 143 * image[1, :, :] + 28 * image[2, :, :] + 186 * image[3, :, :] + 106 * image[4, :, :]

    return RGB_img.astype(np.uint8)


def get_cavf_Sparse_RGBA(image):

    if np.argmin(image.shape) == 2:
        image = image.transpose(2, 0, 1)

    RGBA_img = np.zeros((image.shape[1], image.shape[2], 4))

    RGBA_img[..., 0] = 15 * image[0, ...] + 171 * image[1, ...] + 215 * image[2, ...] +  43 * image[3, ...] + 166 * image[4, ...]
    RGBA_img[..., 1] = 32 * image[0, ...] + 165 * image[1, ...] +  25 * image[2, ...] + 131 * image[3, ...] + 217 * image[4, ...]
    RGBA_img[..., 2] = 53 * image[0, ...] + 143 * image[1, ...] +  28 * image[2, ...] + 186 * image[3, ...] + 106 * image[4, ...]

    # extract the background and the capillaries
    background = np.argmax(image, axis=0) == 0
    capillaries = np.argmax(image, axis=0) == 1
    mask = np.logical_or(background, capillaries)  # Boolean mask

    # mask the backgorund
    RGBA_img[..., 3] = 255

    # Apply mask to set RGB and alpha to 0
    RGBA_img[mask] = [0, 0, 0, 0]


    # Separate RGB and Alpha channels
    rgb = RGBA_img[..., :3]  # Extract RGB channels
    alpha = RGBA_img[..., 3]  # Extract Alpha channel

    # Get the index of the max channel per pixel (ignoring Alpha)
    max_indices = np.argmax(rgb, axis=2)

    # Create an empty RGB array (same shape as input RGB)
    sparse_rgb = np.zeros_like(rgb)

    # Set the max channel to 255
    for i in range(3):  # Iterate over R, G, B channels
        sparse_rgb[..., i] = (max_indices == i) * 255  # Set max channel to 255

    # Combine sparse RGB with original Alpha channel
    sparse_image = np.dstack((sparse_rgb, alpha))  # Stack along last axis to restore RGBA format

    return sparse_image.astype(np.uint8)

def symlink(src, dst):
    if not os.path.exists(src):
        raise ValueError(f"Source does not exist: {src}")

    if os.path.exists(dst):
        os.remove(dst)

    os.symlink(src, dst)


def argparser():
    parser = argparse.ArgumentParser(description='Load cache')
    parser.add_argument('--merged_results_root', type=str, required=True, help='Path to the predictions csv file')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to the cache directory')
    parser.add_argument('--patients_manifest_path', type=str, required=True, help='Path to the patients csv file')
    parser.add_argument('--saveroot', type=str, required=True, help='Path to save directory')
    parser.add_argument('--aireadi_root', type=str, required=True, help='Path to the AIREADI root directory')
    parser.add_argument('--threshold', type=int, default=128)

    # if true will align the size of the final label to the manual label (same size as enface)
    # otherwise will align the size of the final label to the merged_result ((256,256) for octa).
    parser.add_argument('--align_size_to_manual_label', action='store_true', help='Align the size of the enface image to the enface image')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("Loading cache...")
    args = argparser()

    mr_root = args.merged_results_root

    mr_manifest = pd.read_csv(os.path.join(mr_root, 'manifest.tsv'), sep='\t')
    mr_manifest = mr_manifest.rename(columns={'patient_id': 'participant_id'})
    mr_manifest = mr_manifest[mr_manifest['model_type'] == 'OCTA']
    pm_manifest = pd.read_csv(args.patients_manifest_path, sep='\t')
    pm_manifest = pm_manifest.sort_values(by='participant_id')

    pm_manifest = pm_manifest.drop_duplicates(subset=['participant_id', 'manufacturers_model_name',
                                                      'anatomic_region', 'laterality'], keep='last')

    # thresh = 128
    # thresh = 50
    thresh = args.threshold

    df = pd.DataFrame(columns=['participant_id',
                               'manufacturer_model_name',
                               'anatomic_region',
                               'laterality',
                               'label_path',
                               'label_visual_path',
                               'original_enface_path',
                               'original_merged_softmax_path']
    )

    for idx, row in pm_manifest.iterrows():

        id = row['participant_id']
        manufacturer_model_name = row['manufacturers_model_name']
        anatomic_region = row['anatomic_region']
        laterality = row['laterality']


        enface_fp = args.aireadi_root + "/" + row['associated_enface_1_file_path']
        enface_name = os.path.basename(enface_fp)
        path = f'{args.cache_path}/{enface_name}_prediction.png'

        if not os.path.exists(path):
            print(f"Skipping {id} {manufacturer_model_name} {anatomic_region} {laterality} as the manual label does not exist")
            continue

        print(f"Processing {id} {manufacturer_model_name} {anatomic_region} {laterality}")

        merged_result_row = mr_manifest[
            (mr_manifest['participant_id'] == id) &
            (mr_manifest['manufacturer_model_name'] == manufacturer_model_name) &
            (mr_manifest['laterality'] == laterality) &
            (mr_manifest['anatomic_region'] == anatomic_region)
        ]

        if len(merged_result_row) > 1:
            print(merged_result_row)
            raise ValueError(f"Multiple rows found for {id}")





        merged_result_row = merged_result_row.iloc[0]

        softmax_fp = os.path.join(mr_root, merged_result_row['softmax_path'])
        softmax = np.load(softmax_fp)
        A, V, F = 2, 3, 4
        softmax[:, :, 2:] = 0
        argmax = np.argmax(softmax, axis=-1)

        manual_label = np.array(Image.open(path))
        print(manual_label.shape)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(manual_label[550:570, 475:500, 0])
        ax[1].imshow(manual_label[400:600, 400:600, 1])
        ax[2].imshow(manual_label[550:570, 475:500, 2])
        ax[3].imshow(manual_label[550:570, 475:500, 3])
        # fig.colorbar(ax[0].imshow(manual_label[:, :, 0]), ax=ax[0])
        # fig.colorbar(ax[1].imshow(manual_label[:, :, 1]), ax=ax[1])
        # fig.colorbar(ax[2].imshow(manual_label[:, :, 2]), ax=ax[2])
        # fig.colorbar(ax[3].imshow(manual_label[:, :, 3]), ax=ax[3])
        plt.savefig('manual_label.png')
        # print('0', manual_label[550:570, 475:500, 0])
        # # print('1', manual_label[400:600, 400:600, 1])
        # print('2', manual_label[550:570, 475:500, 2])
        # print('3', manual_label[550:570, 475:500, 3])


        manual_label_alpha = manual_label[:, :, 3]

        manual_label = manual_label[:, :, :3]
        # red = np.argmin(manual_label[...,:3], axis=-1) != 0
        manual_label[manual_label_alpha < thresh] = 0
        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(manual_label[550:570, 475:500, 0])
        ax[1].imshow(manual_label[400:600, 400:600, 1])
        ax[2].imshow(manual_label[550:570, 475:500, 2])
        # ax[3].imshow(manual_label[550:570, 475:500, 3])
        plt.savefig('manual_label_thresh.png')
        if not args.align_size_to_manual_label:
            argmax = cv2.resize(argmax, (manual_label.shape[1], manual_label.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            manual_label = cv2.resize(manual_label, (softmax.shape[1], softmax.shape[0]), interpolation=cv2.INTER_NEAREST)
        bg = np.all(manual_label == [0, 0, 0], axis=-1)
        # get artery by checking if the pixel is not in bg and the pixel is largest in the first channel of manual_label
        artery = np.logical_and(~bg, manual_label[..., 0] > manual_label[..., 1])
        artery = np.logical_and(artery, manual_label[..., 0] > manual_label[..., 2])
        # get vein by checking if the pixel is not in bg and the pixel is largest in the third channel of manual_label
        vein = np.logical_and(~bg, manual_label[..., 2] > manual_label[..., 0])
        vein = np.logical_and(vein, manual_label[..., 2] > manual_label[..., 1])

        # get faz by checking if the pixel is not in bg and the pixel is largest in the second channel of manual_label
        faz = np.logical_and(~bg, manual_label[..., 1] > manual_label[..., 0])
        faz = np.logical_and(faz, manual_label[..., 1] > manual_label[..., 2])

        # artery = np.all(manual_label == [255, 0, 0], axis=-1)
        # vein = np.all(manual_label == [0, 0, 255], axis=-1)
        # faz = np.all(manual_label == [0, 255, 0], axis=-1)
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(artery[550:570, 475:500])
        ax[1].imshow(vein[550:570, 475:500])
        ax[2].imshow(faz[550:570, 475:500])
        plt.savefig('manual_label_thresh_new.png')

        mask = np.logical_or(artery, vein)
        mask = np.logical_or(mask, faz)






        merged_manual_label = argmax.copy()
        merged_manual_label[artery] = A
        merged_manual_label[vein] = V
        merged_manual_label[faz] = F
        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(merged_manual_label[550:570, 475:500] == A)
        ax[1].imshow(merged_manual_label[550:570, 475:500] == V)
        ax[2].imshow(merged_manual_label[550:570, 475:500] == F)
        plt.savefig('manual_label_thresh_new_merged.png')
        oh = to_one_hot(merged_manual_label, 5)
        fig, ax = plt.subplots(1, 5)
        ax[0].imshow(oh[550:570, 475:500, 0])
        ax[1].imshow(oh[550:570, 475:500, 1])
        ax[2].imshow(oh[550:570, 475:500, 2])
        ax[3].imshow(oh[550:570, 475:500, 3])
        ax[4].imshow(oh[550:570, 475:500, 4])
        plt.savefig('manual_label_thresh_new_merged_onehot.png')


        merged_manual_label_rgb = get_cavf_RGB(to_one_hot(merged_manual_label, 5))
        fig, ax = plt.subplots(1, 5)
        ax[0].imshow(merged_manual_label_rgb[550:570, 475:500, 0])
        ax[1].imshow(merged_manual_label_rgb[550:570, 475:500, 1])
        ax[2].imshow(merged_manual_label_rgb[550:570, 475:500, 2])
        ax[3].imshow(merged_manual_label_rgb[550:570, 475:500])
        ax[4].imshow(merged_manual_label_rgb)
        plt.savefig('manual_label_thresh_new_merged_rgb.png')

        patient_save_folder = os.path.join(args.saveroot, f'{id}_{laterality}', f'{manufacturer_model_name}_{anatomic_region}')
        os.makedirs(patient_save_folder, exist_ok=True)


        np.save(f'{patient_save_folder}/label.npy', merged_manual_label)
        img = Image.fromarray(merged_manual_label_rgb)
        plt.clf()
        plt.imshow(img)
        plt.savefig('label.png')
        img.save(f'{patient_save_folder}/label.png')
        print(patient_save_folder)




        related_files_folder = os.path.join(patient_save_folder, 'related_files')
        os.makedirs(related_files_folder, exist_ok=True)

        enface_img = pydicom.dcmread(enface_fp).pixel_array
        Image.fromarray(enface_img).save(f'{related_files_folder}/original_enface.png')



        symlink(f'{args.cache_path}/{enface_name}_prediction.png', f'{related_files_folder}/manual_label.png')
        symlink(f'{args.cache_path}/{enface_name}_overlay.png', f'{related_files_folder}/manual_overlay.png')

        symlink(softmax_fp, f'{related_files_folder}/original_merged_softmax.npy')
        symlink(os.path.join(mr_root, merged_result_row['softmax_visual_path']), f'{related_files_folder}/original_merged_softmax.png')


        df = df._append({
            'participant_id': id,
            'manufacturer_model_name': manufacturer_model_name,
            'anatomic_region': anatomic_region,
            'laterality': laterality,
            'label_path': f'{patient_save_folder}/label.npy'.removeprefix(args.saveroot),
            'label_visual_path': f'{patient_save_folder}/label.png'.removeprefix(args.saveroot),
            'original_enface_path': f'{related_files_folder}/original_enface.png'.removeprefix(args.saveroot),
            'original_merged_softmax_path': f'{related_files_folder}/original_merged_softmax.npy'.removeprefix(args.saveroot),
        }, ignore_index=True)

    df.to_csv(os.path.join(args.saveroot, 'manifest.tsv'), sep='\t', index=False)




