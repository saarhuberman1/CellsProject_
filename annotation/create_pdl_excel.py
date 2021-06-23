import os
import glob
import pandas as pd
from tqdm import tqdm

root_dir = r'C:\Users\amirlivn\Downloads\bliss\for_annotation\02-008'

### PDL-1 ###
# pdl1_dir = r'C:\Users\amirlivn\Downloads\bliss\for_annotation\02-008\PDL1(SP142)-Springbio'
# pd_dir = r'C:\Users\amirlivn\Downloads\bliss\for_annotation\02-008\PD1(NAT105)-CellMarque'
# he_dir = r'C:\Users\amirlivn\Downloads\bliss\for_annotation\02-008\HE'

### ER-SP1 ###
pdl1_dir = r'C:\Users\amirlivn\Downloads\bliss\for_annotation\02-008\ER-SP1'
receptor1_name = os.path.basename(pdl1_dir)
pd_dir = None
receptor2_name = 'dont care'
he_dir = r'C:\Users\amirlivn\Downloads\bliss\for_annotation\02-008\HE'

pdl1_images = set([i.replace(root_dir, '') for i in glob.glob(rf'{pdl1_dir}\*.jpg')])
pd_images = set([i.replace(root_dir, '') for i in glob.glob(rf'{pd_dir}\*.jpg')]) if pd_dir is not None else set()
he_images = sorted(list([i.replace(root_dir, '') for i in glob.glob(rf'{he_dir}\*.jpg')]))

he_list = []
pd_list = []
pdl1_list = []
for he_im in tqdm(he_images):
    pd_im = None
    if he_im.replace('HE',receptor2_name) in pd_images:
        pd_im = he_im.replace('HE',receptor2_name)
        pd_images.remove(pd_im)
    pdl1_im = None
    if he_im.replace('HE',receptor1_name).replace('s13','s14') in pdl1_images:
        pdl1_im = he_im.replace('HE',receptor1_name).replace('s13','s14')
        pdl1_images.remove(pdl1_im)
    elif he_im.replace('HE',receptor1_name).replace('_s1_','s14') in pdl1_images:
        pdl1_im = he_im.replace('HE',receptor1_name).replace('_s1_','s14')
        pdl1_images.remove(pdl1_im)
    elif he_im.replace('HE',receptor1_name).replace('_b3_','s14') in pdl1_images:
        pdl1_im = he_im.replace('HE',receptor1_name).replace('_b3_','s14')
        pdl1_images.remove(pdl1_im)


    # for im in pd_images:
    #     if os.path.basename(he_im).replace('HE','PD1(NAT105)-CellMarque') == os.path.basename(he_im):
    #         pd_im = im.replace(root_dir, '')
    #         pd_images.remove(im)
    #         break
    # pdl1_im = None
    # for im in pdl1_images:
    #     if os.path.basename(im).replace('HE','PD1(NAT105)-CellMarque') == os.path.basename(he_im):
    #         pd_im = im.replace(root_dir, '')
    #         pd_images.remove(im)
    #         break
    if pdl1_im is not None or pd_im is not None:
        he_list.append(he_im)
        pd_list.append(pd_im)
        pdl1_list.append(pdl1_im)

print(f'found {len(pdl1_list)} couples')
df = pd.DataFrame({'HE_path': he_list, 'PDL1_path': pdl1_list, 'PDL1_label': [None]*len(pdl1_list), 'PD1_path': pd_list, 'PD1_label': [None]*len(pd_list)})
# df.to_csv(r'C:\Users\amirlivn\Downloads\bliss\for_annotation\PDL1_annotation_task.csv', index=False)
df.to_csv(r'C:\Users\amirlivn\Downloads\bliss\for_annotation\ER-SP1_annotation_task.csv', index=False)


