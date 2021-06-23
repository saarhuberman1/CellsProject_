import shutil
import numpy as np
import os
import glob
import pandas as pd
import cv2


def build_unlabeled_list_from_metadata_file(excel_file, output_dir, balance=False, target_rec='ER', thresh_pos=0.9,
                                            thresh_neg=0.1, none_thresh=750, dup_neg=False):
    root_dir = r'D:\MSc\cancer\Data\data_from_web\bliss'
    ignore_dirs = ['02-008', '01-011']  # , 'CALGB9741', 'CALGB9344', 'NTTMB', '17-007', '17-002', 'CBCTR', 'MA5',
    # 'UKMB09']
    tma_names = [i for i in os.listdir(root_dir) if i not in ignore_dirs]
    if target_rec == 'ER':
        ignore_str = 'HER'
    else:
        ignore_str = 'never'
    os.makedirs(output_dir, exist_ok=True)
    xls = pd.ExcelFile(excel_file)
    df_list = []
    valid_tma_name = []

    none_csv_path = r"D:\cancer\test\log_amir_with_filter.csv"
    root_dir = r'\\ger\ec\proj\ha\RSG\PersonDataCollection1\users\amir\cancer\data\data_from_web\bliss'
    none_df = pd.read_csv(none_csv_path)

    for tma in tma_names:
        try:
            df_list.append(pd.read_excel(xls, tma))
            valid_tma_name.append(tma)
        except:
            continue

    total_lines = []
    total_labels = []
    total_count = 0
    for tab, df in enumerate(df_list):

        print('TMA: {}'.format(valid_tma_name[tab]))
        r_name, r_score1 = None, None
        r_fields = [c for c in df.columns if target_rec in c and ignore_str not in c]
        if 'Blaise1TMA' in valid_tma_name[tab]:
            flag = 0
        if len(r_fields) > 2:
            print('Error: found to many receptors columns: {}'.format(r_fields))
            continue
        elif len(r_fields) == 0:
            print('No receptors images for tma: ', valid_tma_name[tab])
            continue
        else:
            [r_name, r_score1] = r_fields
            print(f'found field: {r_score1}')


        images = []
        labels = []
        for i, row in df.iterrows():
            imagename = row[r_name]
            if pd.isna(imagename):
                continue
            image_path = fr'{root_dir}/{valid_tma_name[tab]}/{r_name}/{imagename}'.replace('/', '\\')
            if not os.path.exists(image_path):
                continue
            total_count += 1
            num_cells = int(none_df.loc[none_df['src_path'] == image_path]['num_cells'])
            if num_cells < none_thresh:
                continue
            if isinstance(row[r_score1], str):
                score = np.average([float(s) for s in row[r_score1].split(',')])
            else:
                score = row[r_score1]
            if not thresh_neg < score < thresh_pos:
                he_imagename = row['HE']
                he_image_path = fr'{root_dir}/{valid_tma_name[tab]}/HE/{he_imagename }'.replace('/', '\\')
                num_he_cells = int(none_df.loc[none_df['src_path'] == he_image_path]['num_cells'])
                if num_he_cells < none_thresh:
                    continue
                images.append(row['HE'])
                label = 0 if score <= thresh_neg else 1
                labels.append(label)

        if len(images) == 0:
            continue

        lines = ['/{}/HE/{} {}\n'.format(valid_tma_name[tab], n, l) for (n, l) in zip(images, labels)]
        num0 = len([i for i in labels if i == 0])
        num1 = len([i for i in labels if i == 1])
        total = len(labels)
        print('label 0 - {}, label 1 - {}. total of {} out of {} images ({}%)'.
              format(num0, num1, total, len(df), int(100 * total / len(df))))
        print('----------------')
        total_lines += lines
        total_labels += labels

    num0 = len([i for i in total_labels if i == 0])
    num1 = len([i for i in total_labels if i == 1])
    total_num = num0 + num1
    precent = int(100 * total_num / total_count)
    print('\n\n')
    print('found total of {} images to use ({}%)! label 0: {}, label 1: {}'.format(len(total_lines), precent, num0,
                                                                                   num1))

    if balance:
        ratio = int(np.ceil((len([i for i in total_labels if i == 1]) / len([i for i in total_labels if i == 0]))))
        if ratio > 1:
            ratio = max([ratio - 1, 0])
            total_lines += ratio * [l for l in total_lines if int(l.strip().split()[1]) == 0]
        elif ratio > 0:
            ratio = int(np.ceil(1 / ratio))
            ratio = max([ratio - 1, 0])
            total_lines += ratio * [l for l in total_lines if int(l.strip().split()[1]) != 0]
    print('balanced: label 0 - {}, label 1 - {}'.
          format(len([i for i in total_lines if int(i.strip().split()[1]) == 0]),
                 len([i for i in total_lines if int(i.strip().split()[1]) == 1])))
    if dup_neg:
        total_lines += [l for l in total_lines if int(l.strip().split()[1]) == 0]
        print('dup_neg: label 0 - {}, label 1 - {}'.
              format(len([i for i in total_lines if int(i.strip().split()[1]) == 0]),
                     len([i for i in total_lines if int(i.strip().split()[1]) == 1])))

    print('\n\n')

    test_list_path = r'{}\{}_{}_{}_{}_fold1.txt'.format(output_dir, target_rec, thresh_neg, thresh_pos, none_thresh)
    if dup_neg:
        test_list_path = test_list_path.replace('.txt', '_dup_neg.txt')
    with open(test_list_path, 'w') as f:
        f.writelines(total_lines)


def build_labeled_list_from_metadata_file(excel_file, output_dir, balance=False, none_excel=None):
    tma_names = ['01-011', '02-008']
    # tma_names = ['02-008']
    # tma_names = ['01-011']
    os.makedirs(output_dir, exist_ok=True)
    xls = pd.ExcelFile(excel_file)
    df_list = []
    for tma in tma_names:
        df_list.append(pd.read_excel(xls, tma))
    df = pd.concat(df_list)

    none_df_list = []
    if none_excel is not None:
        xls_none = pd.ExcelFile(none_excel)
        for tma in tma_names:
            none_df_list.append(pd.read_excel(xls_none, tma))
        df_none = pd.concat(none_df_list)
        none_images = set(list(df_none['im_name']))
    else:
        none_images = []

    for fold in range(1, 8):
        test_data = df.loc[df['fold'] == fold]
        train_data = df.loc[(df['fold'] != fold) & (df['fold'] != 'None')]

        test_images = list(test_data['HE'])
        test_labels = [i if i == 0 else 1 for i in list(test_data['ER_label'])]
        test_ids = list(test_data['id'])
        train_images = list(train_data['HE'])
        train_labels = [i if i == 0 else 1 for i in list(train_data['ER_label'])]
        train_ids = list(train_data['id'])

        test_lines = ['{} {} {}\n'.format(n, l, int(i)) for (n, l, i) in zip(test_images, test_labels, test_ids) if
                      '02-008' in n]
        train_lines = ['{} {} {}\n'.format(n, l, int(i)) for (n, l, i) in zip(train_images, train_labels, train_ids)
                       if os.path.basename(n) not in none_images]

        if balance:
            ratio = int(len([i for i in train_labels if i == 1]) / len([i for i in train_labels if i == 0]))
            ratio = max([ratio - 1, 0])
            train_lines += ratio * [l for l in train_lines if int(l.strip().split()[1]) == 0]
            train_labels = [int(l.strip().split()[1]) for l in train_lines]
        print('Test Fold: {}:'.format(fold))
        print('test set: label 0 - {}, label 1 - {}'.format(len([i for i in test_labels if i == 0]),
                                                            len([i for i in test_labels if i == 1])))
        print('train set: label 0 - {}, label 1 - {}'.format(len([i for i in train_labels if i == 0]),
                                                             len([i for i in train_labels if i == 1])))
        print('\n\n')

        test_list_path = r'{}\fold{}_test.txt'.format(output_dir, fold)
        train_list_path = r'{}\fold{}_train.txt'.format(output_dir, fold)
        with open(test_list_path, 'w') as f:
            f.writelines(test_lines)
        with open(train_list_path, 'w') as f:
            f.writelines(train_lines)


def build_receptors_list_from_metadata_file(excel_file, output_dir, balance=False):
    tma_names = ['01-011', '02-008', '08-006']
    os.makedirs(output_dir, exist_ok=True)
    xls = pd.ExcelFile(excel_file)
    df_list = []
    for tma in tma_names:
        df_list.append(pd.read_excel(xls, tma))
    df = pd.concat(df_list)
    receptors = ['EGFR', 'EGFRpharmDx-Dako', 'ER', 'ER-SP1', 'ER-VGH', 'PR', 'Ki67-Neo', 'RET']  # if tma
    # == '02-008'
    # else ["ER"]
    for fold in range(1, 7):
        test_data = df.loc[df['fold'] == fold]
        train_data = df.loc[(df['fold'] != fold) & (df['fold'] != 'None')]
        test_images, test_labels, train_images, train_labels = [], [], [], []
        for r in receptors:
            test_images += [i.replace('\\', '/') for i in list(test_data[f'{r}']) if
                            '[' not in str(i) and str(i) != 'nan']
            test_labels += [i for i in list(test_data[f'{r}_label']) if '[' not in str(i) and str(i) != 'nan']
            train_images += [i.replace('\\', '/') for i in list(train_data[f'{r}']) if
                             '[' not in str(i) and str(i) != 'nan']
            train_labels += [i for i in list(train_data[f'{r}_label']) if '[' not in str(i) and str(i) != 'nan']

        test_lines = ['{} {}\n'.format(n, int(l)) for (n, l) in zip(test_images, test_labels) if
                      str(l) != 'None' and str(l) != 'nan']
        train_lines = ['{} {}\n'.format(n, int(l)) for (n, l) in zip(train_images, train_labels) if
                       str(l) != 'None' and str(l) != 'nan']

        if balance:
            ratio = int(len([i for i in train_labels if i != 0]) / len([i for i in train_labels if i == 0]))
            if ratio > 1:
                ratio = max([ratio - 1, 0])
                train_lines += ratio * [l for l in train_lines if int(l.strip().split()[1]) == 0]
            elif ratio > 0:
                ratio = int(1 / ratio)
                ratio = max([ratio - 1, 0])
                train_lines += ratio * [l for l in train_lines if int(l.strip().split()[1]) != 0]

        train_labels = [int(l.strip().split()[1]) for l in train_lines]
        print('Test Fold: {}:'.format(fold))
        print('test set: label 0 - {}, label 1 - {}'.format(len([i for i in test_labels if i == 0]),
                                                            len([i for i in test_labels if i != 0])))
        print('train set: label 0 - {}, label 1 - {}'.format(len([i for i in train_labels if i == 0]),
                                                             len([i for i in train_labels if i != 0])))
        print('\n\n')

        test_list_path = r'{}\fold{}_test.txt'.format(output_dir, fold)
        train_list_path = r'{}\fold{}_train.txt'.format(output_dir, fold)
        with open(test_list_path, 'w') as f:
            f.writelines(test_lines)
        with open(train_list_path, 'w') as f:
            f.writelines(train_lines)


def validate_list(list_path, root_dir):
    with open(list_path, 'r') as f:
        lines = f.readlines()

    done_images = set()
    valid_lines = []
    bad = 0
    for line in lines:
        im = line.strip().split()[0]
        if im in done_images:
            continue

        read_test = cv2.imread(r'{}\{}'.format(root_dir, im), cv2.IMREAD_UNCHANGED)
        if read_test is not None:
            valid_lines.append(line)
        else:
            bad += 1

        done_images.add(im)

    print('removed {} invalid images from list: {}'.format(bad, list_path))
    with open(list_path, 'w') as f:
        f.writelines(valid_lines)


def compare_lists(list1, list2):
    with open(list1, 'r') as f:
        lines1 = f.readlines()
        images1 = set([l.strip().split()[0] for l in lines1])
    with open(list2, 'r') as f:
        lines2 = f.readlines()
        images2 = set([l.strip().split()[0] for l in lines2])
    for im_path in images1:
        if im_path in images2:
            print(im_path)
            return
    print('lists are distinct!')


def copy_files_by_list(train_list, root_dir, target_dir):
    with open(train_list, 'r') as f:
        images = [l.strip().split()[0] for l in f.readlines()]

    for src in images:
        dst = src.replace(root_dir,target_dir)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        else:
            print(dst)
            return


if __name__ == '__main__':
    # unlabeled data
    # metadata_file = r"D:\MSc\cancer\Data\data_from_web\bliss_metadata_unlabeled_17.1.21_with_score.xlsx"
    # output_dir = r'D:\MSc\cancer\Data\data_from_web\lists\HE_unlabeled_new'

    # labeled data
    metadata_file = r"D:\MSc\cancer\Data\data_from_web\bliss_metadata_labeled_23.2.xlsx"
    output_dir = r'D:\MSc\cancer\Data\data_from_web\lists\labeled_receptors'

    os.makedirs(output_dir,exist_ok=True)
    none_excel = r"D:\MSc\cancer\Data\data_from_web\suspected_none\suspected_None_images.xlsx"
    build_labeled_list_from_metadata_file(metadata_file, output_dir, balance=True, none_excel=none_excel)

    # build_receptors_list_from_metadata_file(metadata_file, output_dir, balance=True)
    # build_unlabeled_list_from_metadata_file(metadata_file, output_dir, balance=True, target_rec='ER', thresh_pos=0.7, thresh_neg=0.3)

