import copy
import urllib.request
# import urllib3.request
from io import BytesIO
from PIL import Image
import numpy as np
import os
from multiprocessing import Pool
import urllib3

def par_copy_im(data):
    blank_image = Image.new("RGB", (2256, 1440))
    # (img1, i, j) = data
    (URL, i, j) = data
    file = BytesIO(urllib.request.urlopen(URL + str(i * 9 + j) + ".jpg", timeout=200).read())
    # file = BytesIO(urllib3.request.urlopen(URL + str(i * 9 + j) + ".jpg").read())
    img1 = Image.open(file)
    if j == 0:
        blank_image.paste(img1, (0, 0))
    elif j == 1:
        blank_image.paste(img1, (752, 0))
    elif j == 2:
        blank_image.paste(img1, (752 * 2, 0))
    elif j == 3:
        blank_image.paste(img1, (752 * 2, 480))
    elif j == 4:
        blank_image.paste(img1, (752, 480))
    elif j == 5:
        blank_image.paste(img1, (0, 480))
    elif j == 6:
        blank_image.paste(img1, (0, 480 * 2))
    elif j == 7:
        blank_image.paste(img1, (752, 480 * 2))
    elif j == 8:
        blank_image.paste(img1, (752 * 2, 480 * 2))
    return blank_image

def download_im(URL, i, blank_image):
    # blank_image = Image.new("RGB", (2256, 1440))
    # (URL, i, j) = data
    for j in range(9):
        file = BytesIO(urllib.request.urlopen(URL + str(i * 9 + j) + ".jpg", timeout=200).read())
        # file = BytesIO(urllib3.request.urlopen(URL + str(i * 9 + j) + ".jpg").read())
        img1 = Image.open(file)
        if j == 0:
            blank_image.paste(img1, (0, 0))
        elif j == 1:
            blank_image.paste(img1, (752, 0))
        elif j == 2:
            blank_image.paste(img1, (752 * 2, 0))
        elif j == 3:
            blank_image.paste(img1, (752 * 2, 480))
        elif j == 4:
            blank_image.paste(img1, (752, 480))
        elif j == 5:
            blank_image.paste(img1, (0, 480))
        elif j == 6:
            blank_image.paste(img1, (0, 480 * 2))
        elif j == 7:
            blank_image.paste(img1, (752, 480 * 2))
        elif j == 8:
            blank_image.paste(img1, (752 * 2, 480 * 2))
        # return blank_image


if __name__ == '__main__':

    # meta_data_file = r"C:\Users\amirlivn\Downloads\bliss\metadata_01.txt"
    meta_data_file = r"C:\Users\amirlivn\Downloads\bliss\metadata_PDL1.txt"
    target_root = r'C:\Users\amirlivn\Downloads\bliss\bliss'
    with open(meta_data_file,'r') as f:
        lines = f.readlines()

    bliss_folder_list = []
    prefix_list = []
    Stain_name_list = []
    block_list_list = []
    stain_names = []
    stain_blocks = []
    for line in lines:
        line = line.strip()
        if 'END' in line:
            break
        if 'bliss_folder' in line:
            bliss_folder_list.append(line.split()[-1])
        elif 'prefix' in line:
            prefix_list.append(line.split()[-1])
        elif 'Stain_name' in line:
            stain_names.append(line.split()[-1])
        elif 'block_list' in line:
            stain_blocks.append(line.split()[-1].split(','))
        else:
            Stain_name_list.append(copy.deepcopy(stain_names))
            block_list_list.append(copy.deepcopy(stain_blocks))
            stain_names = []
            stain_blocks = []


    for (bliss_folder,prefix,stain_names,blocks_list) in zip(bliss_folder_list,prefix_list,
                                                                       Stain_name_list, block_list_list):
        for Stain_name, block_list in zip(stain_names, blocks_list):
            folder_output = '{}\{}\{}'.format(target_root,bliss_folder,Stain_name)
            print('Processing stain: {}'.format(Stain_name))
            print('output dir: {}'.format(folder_output))
            os.makedirs(folder_output, exist_ok=True)
            folder_output += '\\'

            if not os.path.isdir(folder_output):
                print('creating folder...')
                os.makedirs(folder_output)

            website = "http://bliss.gpec.ubc.ca/WebSlides/"

            blank_image = np.asarray(Image.new("RGB", (2256, 1440)))

            # for block_name in block_list[20:]:
            for block_name in block_list:
                TMA_name = Stain_name + '_' + block_name
                print(f'Starting with TMA {TMA_name} ...\n')
                URL = website + '_' + prefix + '_' + bliss_folder + "_" + TMA_name + "/Da"
                i = 0
                while True:
                    out_file = TMA_name + "_%03d" % i + ".jpg"
                    if os.path.exists(folder_output + out_file):
                        # print(out_file + ' exists. skipping...')
                        i += 1
                        continue

                    try:  # check if URL exists
                        print(URL + str(i * 9) + ".jpg")
                        urllib.request.urlopen(URL + str(i * 9) + ".jpg", timeout=200)
                        # urllib3.request.urlopen(URL + str(i * 9) + ".jpg")
                        url_im_path = URL + str(i * 9) + ".jpg"
                        # http = urllib3.PoolManager()
                        # r = http.request('GET', url_im_path)
                        # img_data = r.data
                        flag = 0

                    except Exception as e:
                        print(e)
                        print(f'\nFinished with TMA {TMA_name} ...\n')
                        # assert i > 0, "no valid urls in TMA_name"
                        break
                    try:
                        blank_image = np.asarray(Image.new("RGB", (2256, 1440)))
                        data_for_par = [(URL, i, j) for j in range(9)]
                        # data_for_par = [(img_data, i, j) for j in range(9)]
                        with Pool(processes=2) as pool:
                            # pool.map(par_copy_im, data_for_par)
                            for im in pool.imap(par_copy_im,data_for_par):
                                blank_image = blank_image + np.asarray(im)

                        blank_image = Image.fromarray(blank_image)

                        blank_image.save(folder_output + out_file, format="JPEG", quality=90)
                        print('saving ' + folder_output + out_file)
                    except Exception as e:
                        blank_image = np.asarray(Image.new("RGB", (2256, 1440)))
                        print(e)
                        pass
                    i += 1
