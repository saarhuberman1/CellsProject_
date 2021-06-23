import pandas as pd
import os
import shutil
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
import sys
import math


def load_labels_data(tma_name, use_ID=True):
    meta_data_file = r"D:\MSc\cancer\Data\data_from_web\TMA_MetaData.xlsx"
    xls = pd.ExcelFile(meta_data_file)
    df = pd.read_excel(xls, '{}_per_image'.format(tma_name))

    id_dict = {}
    # target_receptors = ['ER', 'ER-SP1', 'ER-VGH', 'PR', 'HER2-SP3', 'HER2-VGH', 'Ki67-Neo', 'EGFR', 'HER2-SP3',
    #                     'PR-Ventana', 'RET']
    if tma == '01-011':
        target_receptors = ['ER']
    elif tma == '02-008':
        target_receptors = ['ER-SP1', 'PR', 'HER2-VGH', 'Ki67-Neo', 'EGFR', 'HER2-SP3', 'RET']
    elif tma == '08-006':
        target_receptors = ['ER-VGH', 'HER2-SP3', 'PR-Ventana']

    # target_receptors = ['RET']
    HE_name_dict = {}
    # get id for each image:
    if use_ID:
        try:
            df_per_id = pd.read_excel(xls, '{}_per_ID'.format(tma_name))
        except:
            df_per_id = pd.read_excel(xls, '{}_per_image'.format(tma_name))

        for i,row in tqdm(df_per_id.iterrows(), desc='reading ID indexes per image'):
            ID = row['PatientID']
            he_names = [str(n) for n in row if 'HE' in str(n)]
            subject_label_dict = {'ID': ID}
            for tr in target_receptors:
                try:
                    label = row[tr]
                    if np.isnan(label) or str(label) == 'nan' or str(label) == 'NaN':
                        label = 'None'
                    else:
                        label = int(label)
                    subject_label_dict[tr] = label
                except:
                    pass
            for name in he_names:
                try:
                    _, letter_idx, v, s, img_num, _ = name.split('_')
                    if s != 'b3':
                        img_num = int(img_num) - 1
                    else:
                        img_num = int(img_num)
                except:
                    _, letter_idx, v, s, img_num = name.split('_')
                    img_num = int(img_num)
                key = (letter_idx.replace('34','3').replace('3','34').replace('12','1').replace('1','12'), v, s,
                       img_num)
                id_dict[key] = subject_label_dict

    metadata_dict = {}
    tf_dict = {}
    ids = list(set(df['PatientID']))
    random.shuffle(ids)
    for i, row in tqdm(df.iterrows(), desc='reading {} labeled annotations'.format(tma_name)):
        name = row['H&E ImageName']
        if str(name) == 'nan':
            continue
        try:
            _, letter_idx, v, s, img_num, _ = name.split('_')
            img_num = int(img_num) - 1
        except:
            _, letter_idx, v, s, img_num = name.split('_')
            img_num = int(img_num)

        key = (letter_idx.replace('34', '3').replace('3', '34').replace('12', '1').replace('1', '12'), v, s,
               img_num)
        metadata_dict[key] = {}
        HE_name_dict[key] = name

        try:
            tf = row['TestFold']
        except Exception as e:
            # print(e)
            # raise Exception(e)
            tf = int(ids.index(row['PatientID'])) % 6 + 1
        if str(tf) == 'nan':
            tf = 'None'
        tf_dict[key] = tf

        for tr in target_receptors:
            try:
                label = row[tr]
                if np.isnan(label) or str(label) == 'nan' or str(label) == 'NaN':
                    label = 'None'
                else:
                    label = int(label)
                if use_ID and label != id_dict[key][tr]:
                    raise Exception('Err! mismatch between per id label and per image label')
                metadata_dict[key][tr] = label
            except Exception as e:
                continue

    return metadata_dict, id_dict, tf_dict, HE_name_dict

if __name__ == '__main__':

    root_dir = r'D:\MSc\cancer\Data\data_from_web\bliss'

    TMA_dirs = ['02-008', '01-011', '08-006']
    # TMA_dirs = ['01-011', '08-006']
    # TMA_dirs = ['08-006']

    use_ID = True

    total_dict = {}

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    metadata_file = r'D:\MSc\cancer\Data\data_from_web\bliss_metadata_labeled_23.2.xlsx'
    print(r'writing results to: {}'.format(metadata_file))
    with pd.ExcelWriter(metadata_file) as writer:

        for tma in TMA_dirs:
            labels_dict, id_dict, tf_dict, HE_name_dict = load_labels_data(tma, use_ID)
            f1_keys = set([k for k, v in tf_dict.items() if v == 1])
            done_keys = set()
            try:
                tma_metadata = {}

                tma_dir = r'{}\{}'.format(root_dir, tma)
                receptors = os.listdir(tma_dir)
                # receptors = ['ER-SP1']
                # if 'HE' not in receptors:
                #     print('ERR! no HE dir in tma: {}'.format(tma_dir))

                receptors = [i for i in receptors if i != 'HE']

                receptors_dict = {}
                for r in receptors:
                    r_dir = r'{}/{}'.format(tma_dir, r)
                    r_images = glob.glob(r'{}/*.jpg'.format(r_dir))
                    receptors_dict[r] = r_images
                HE_dir = r'{}/HE'.format(tma_dir)
                HE_images = glob.glob(r'{}/*.jpg'.format(HE_dir))
                # HE_images = [i for i in HE_images if 'A3_v3_b3' in i]

                sheet_df = {'HE': [], 'HE_orig_format': [], 'TMA': [], 'fold': [], 'id': []}
                for r in receptors:
                    sheet_df[r] = []
                    sheet_df[r + '_label'] = []

                for he_im in tqdm(HE_images, desc='analyzsing TMA: {}'.format(tma)):


                    he_name = os.path.basename(he_im)

                    he_name_split = he_name.split('_')
                    he_letter_idx = he_name_split[1].replace('34', '3').replace('3', '34').replace('12', '1').replace(
                        '1', '12')
                    he_img_num = int(he_name_split[-1].replace('.jpg', ''))

                    v, s = he_name_split[2:4]
                    he_key = (he_letter_idx, v, s, he_img_num)
                    try:
                        orig_name = HE_name_dict[he_key]
                        he_fold = tf_dict[he_key]
                        if use_ID:
                            subject_id = id_dict[he_key]['ID']
                        else:
                            subject_id = -1
                        done_keys.add(he_key)
                    except:
                        continue


                    sheet_df['HE'].append(he_im.replace(root_dir,'').replace('\\', '/'))
                    sheet_df['HE_orig_format'].append(orig_name)
                    sheet_df['TMA'].append(tma)
                    sheet_df['fold'].append(he_fold)
                    sheet_df['id'].append(subject_id)

                    for r in receptors:
                        try:
                            label = labels_dict[he_key][r]
                            if np.isnan(label) or str(label) == 'nan' or str(label) == 'NaN':
                                label = 'None'
                            else:
                                label = int(label)
                            if use_ID and label != id_dict[he_key][r]:
                                raise Exception('Err! mismatch between per id label and per image label')
                        except:
                            label = 'None'
                        r_images = []
                        r_paths = []
                        r_label = []
                        for r_im in receptors_dict[r]:
                            try:
                                r_name = os.path.basename(r_im)
                                r_name_split = r_name.split('_')
                                r_letter_idx = r_name_split[1].replace('34', '3').replace('3', '34').replace('12', '1').replace(
                        '1', '12').replace('12','2').replace('2','12')
                                r_img_num = int(r_name_split[-1].replace('.jpg', ''))

                                if r_letter_idx == he_letter_idx and r_img_num == he_img_num:
                                    r_images.append(r_name)
                                    r_paths.append(r_im.replace(root_dir,''))
                                    r_label.append(label)
                            except Exception as e:
                                print(e)
                                continue

                        if len(r_images) == 0:
                            r_images = 'None'
                            r_paths = 'None'
                            r_label = 'None'
                        elif len(r_images) == 1:
                            r_images = r_images[0]
                            r_paths = r_paths[0]
                            r_label = r_label[0]

                        sheet_df[r].append(r_paths)
                        sheet_df[r + '_label'].append(r_label)
                print(f1_keys - done_keys)
                # Write each dataframe to a different worksheet.
                he_num = len(sheet_df['HE'])
                sheet_df_to_write = {k: v for k,v in sheet_df.items() if len(v) == he_num}
                df = pd.DataFrame(sheet_df_to_write)
                df.to_excel(writer, sheet_name=tma, index=False)
            except Exception as e:
                print('Error in TMA: {}'.format(tma))
                print(e)
                print('\n')
                continue
    flag = 0
