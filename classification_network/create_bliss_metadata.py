import json
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


def get_r_type(r):
    r = r.lower().split('-')[0]
    if 'thermo' in r:
        r = r.replace('thermo', '')

    if 'her2' in r:
        return 'HER2'
    elif 'er' in r:
        return 'ER'
    elif 'ki67' in r or 'ki-67' in r:
        return 'Ki67'
    elif 'pdl' in r:
        return 'PDL1'
    elif 'egfr' in r:
        return 'EGFR'
    elif 'pr' in r:
        return 'PR'
    else:
        raise Exception(f'unkown receptor {r}')


def im_parmas(im_path):
    tma = os.path.dirname(im_path).split('\\')[-1].split('/')[0]
    name = os.path.basename(im_path)
    name_split = name.split('_')
    if tma in ['CALGB9344','CALGB9741','CBCTR','MA31','MPNST','NTTMB','UKMB09']:
        if tma in ['MA31','MPNST', 'NTTMB', 'UKMB09']:
            letter_idx = name_split[1]
        else:
            letter_idx = name_split[1] + '_' + name_split[2]
    else:
        letter_idx = name_split[1].replace('34', '3').replace('3', '34').replace('12', '1').replace('1', '12')
    img_num = int(name_split[-1].replace('.jpg', ''))
    return name, letter_idx, img_num


if __name__ == '__main__':
    # root_dir = r'D:\MSc\cancer\Data\data_from_web\bliss'
    root_dir = r'\\ger\ec\proj\ha\RSG\PersonDataCollection1\users\amir\cancer\data\data_from_web\bliss'
    # TMA_dirs = ['03-005']
    save_score = True
    ignore_dirs = ['02-008', '01-011']
    TMA_dirs = [i for i in os.listdir(root_dir) if i not in ignore_dirs]
    # TMA_dirs = ['14-004']
    total_dict = {headline: [] for headline in ['TMA', 'HE - total', 'HE<->ER', 'HE<->HER2', 'HE<->Ki67', 'HE<->EGFR',
                                                'HE<->PR', 'HE<->PDL1']}
    HEAD_receptors = {'ER', 'HER2', 'Ki67', 'EGFR', 'PR', 'PDL1'}

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    metadata_file = r'D:\MSc\cancer\Data\data_from_web\bliss_metadata_unlabeled_17.1.21.xlsx'
    if save_score:
        metadata_file = metadata_file.replace('.xlsx', '_with_score.xlsx')

    none_csv_path = r"D:\cancer\test\log_amir_with_filter.csv"
    root_dir = r'\\ger\ec\proj\ha\RSG\PersonDataCollection1\users\amir\cancer\data\data_from_web\bliss'
    none_df = pd.read_csv(none_csv_path)

    print(r'writing results to: {}'.format(metadata_file))
    with pd.ExcelWriter(metadata_file) as writer:

        for tma in TMA_dirs:
            try:
                tma_metadata = {}

                tma_dir = r'{}\{}'.format(root_dir, tma)
                receptors = os.listdir(tma_dir)

                if 'HE' not in receptors:
                    print('ERR! no HE dir in tma: {}'.format(tma_dir))

                receptors = [i for i in receptors if i != 'HE']
                # receptors = [i for i in receptors if i != 'HE' and 'ER' in i and 'HER' not in i]

                receptors_dict = {}
                for r in receptors:
                    r_dir = r'{}/{}'.format(tma_dir, r)

                    # handle egde cases:
                    if 'PDL1[' in r_dir:
                        r_dir = r_dir.split('PDL1[')[0] + 'PDL*'
                    elif 'ER[' in r_dir:
                        r_dir = r_dir.split('ER[')[0] + 'ER*'
                    elif 'HER2{' in r_dir:
                        r_dir = r_dir.split('HER2{')[0] + 'HER2*'
                    elif 'Ki67[' in r_dir:
                        r_dir = r_dir.split('Ki67[')[0] + 'Ki67*'
                    elif 'EGFR[' in r_dir:
                        r_dir = r_dir.split('EGFR[')[0] + 'EGFR*'
                    elif 'PR[' in r_dir:
                        r_dir = r_dir.split('PR[')[0] + 'PR*'

                    r_images = glob.glob(r'{}/*.jpg'.format(r_dir))
                    if get_r_type(r) == 'ER':
                        r_images = [i for i in r_images if 'HER2' not in i]
                    receptors_dict[r] = r_images
                HE_dir = r'{}/HE'.format(tma_dir)
                HE_images = glob.glob(r'{}/*.jpg'.format(HE_dir))

                total_dict['TMA'].append(tma)
                total_dict['HE - total'].append(len(HE_images))

                sheet_df = {'HE': [], 'TMA': [], 'HE_cell_count': []}

                for r in receptors:
                    sheet_df[r] = []
                    sheet_df[r + '_cell_count'] = []
                    if save_score:
                        sheet_df[r + '_fold1_score_eb5'] = []

                for he_im in tqdm(HE_images, desc='analyzsing TMA: {}'.format(tma)):

                    tma_metadata[he_im] = {r: [] for r in receptors}
                    he_name, he_letter_idx, he_img_num = im_parmas(he_im)
                    none_score = int(none_df.loc[none_df['src_path'] == he_im.replace('/','\\')]['num_cells'])
                    sheet_df['HE'].append(he_name)
                    sheet_df['HE_cell_count'].append(none_score)
                    sheet_df['TMA'].append(tma)



                    for r in receptors:
                        r_images = []
                        r_paths = []
                        r_1scores = []
                        r_none_score = []
                        for r_im in receptors_dict[r]:
                            try:
                                r_name, r_letter_idx, r_img_num = im_parmas(r_im)
                                if r_letter_idx == he_letter_idx and r_img_num == he_img_num:
                                    r_json = r_im.replace('.jpg','.json')
                                    if os.path.exists(r_json):
                                        with open(r_json,'r') as f:
                                            fold_label = json.load(f)
                                        # r_1scores.append(fold_label['fold1_score'])
                                        r_1scores.append(fold_label['fold1_score_eb5'])
                                    else:
                                        r_1scores.append(-1)

                                    none_score = int(none_df.loc[none_df['src_path'] == r_im.replace('/','\\')]['num_cells'])

                                    r_images.append(r_name)
                                    r_paths.append(r_im)
                                    r_none_score.append(none_score)
                                    tma_metadata[he_im][r].append(r_im)
                            except Exception as e:
                                print(e)
                                continue

                        if len(r_images) == 0:
                            r_images = None
                            r_paths = None
                            r_1scores = None
                            r_none_score = None
                        elif len(r_images) == 1:
                            r_images = r_images[0]
                            r_paths = r_paths[0]
                            r_1scores = r_1scores[0]
                            r_none_score = r_none_score[0]
                        else:
                            r_images = ','.join(r_images)
                            r_paths = ','.join(r_paths)
                            r_1scores = ','.join([str(s) for s in r_1scores])
                            r_none_score = ','.join([str(s) for s in r_none_score])

                        sheet_df[r].append(r_images)
                        sheet_df[r+'_cell_count'].append(r_none_score)
                        # sheet_df[r + '_fold1_score'].append(r_1scores)
                        if save_score:
                            sheet_df[r + '_fold1_score_eb5'].append(r_1scores)

                done_set = set()
                count_dict = {r: 0 for r in set([get_r_type(r) for r in receptors])}
                for i,r in enumerate(receptors):
                    r_type = get_r_type(r)
                    num_images = len([i for i in sheet_df[r] if i is not None])
                    count_dict[r_type] += num_images
                    done_set.add(r_type)

                for r_type in count_dict.keys():
                    total_dict[f'HE<->{r_type}'].append(count_dict[r_type])

                # complete missing receptors:
                for r_type in HEAD_receptors - done_set:
                    total_dict[f'HE<->{r_type}'].append(0)

                # Write each dataframe to a different worksheet.
                df = pd.DataFrame(sheet_df)
                df.to_excel(writer, sheet_name=tma, index=False)
            except Exception as e:
                print('Error in TMA: {}'.format(tma))
                print(e)
                print('\n')
                continue

        df = pd.DataFrame(total_dict)
        heads = ['HE<->ER', 'HE<->HER2', 'HE<->Ki67', 'HE<->EGFR',
                                                'HE<->PR', 'HE<->PDL1']
        mis_list = []
        for i,row in df.iterrows():
            missing = row['HE - total'] - np.sum([row[k] for k in heads])
            mis_list.append(missing)
        df['HE - unpaired'] = mis_list
        df.to_excel(writer, sheet_name='Summary', index=False)

    flag = 0
