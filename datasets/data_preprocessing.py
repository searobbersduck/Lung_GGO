import os
import sys
import pandas as pd
from tqdm import tqdm
from glob import glob

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(root)
sys.path.append(root)
from external_lib.MedCommon.utils.data_io_utils import DataIO
from external_lib.MedCommon.utils.anno_analyze_utils import DetectionAnnotationAnalyzeUtils
from external_lib.MedCommon.utils.mask_bounding_utils import MaskBoundingUtils

import SimpleITK as sitk

data_root = '/data/medical/hospital/cz/ggo'

def extract_block_cz_pos(block_size):
    anno_file = '/data/medical/hospital/cz/ggo/annotation/标注结果-长征非测试集(阳性）/01回流数据信息表（UID)-DD1209V6- 阳（阴）性含测试集数据数量表格.xlsx'
    df = pd.read_excel(anno_file, sheet_name=1)
    
    in_root = os.path.join(data_root, 'cz/raw/pos/images')
    out_root = os.path.join(data_root, 'cz', 'block_{}'.format(block_size),'pos')
    os.makedirs(out_root, exist_ok=True)

    center_col_name = ['coordX', 'coordY', 'coordZ']
    suid_dict = {}
    for index, row in df.iterrows():
        coord_center = row[center_col_name]
        suid = row['series uid'].strip()
        if suid not in suid_dict:
            suid_dict[suid] = 1
        else:
            suid_dict[suid] += 1
        image_file = os.path.join(in_root, suid)
        try:
            image = DataIO.load_dicom_series(image_file)['sitk_image']
            pix_coord_center = DetectionAnnotationAnalyzeUtils.PysicalCoordinate2PixelCoordinate(coord_center, image)
            image_block = DetectionAnnotationAnalyzeUtils.extract_block_around_center_image(image, pix_coord_center, block_size)
            
            idx = '{}_{}'.format(suid, suid_dict[suid])
            out_sub_dir = os.path.join(out_root, idx)
            os.makedirs(out_sub_dir, exist_ok=True)
            out_file = os.path.join(out_sub_dir, 'block.nii.gz'.format(block_size))
            sitk.WriteImage(image_block, out_file)
        except Exception as e:
            print('====> Error case:\t', row['series uid'])
            print(e)

def extract_block_cz_neg(block_size):
    anno_file = '/data/medical/hospital/cz/ggo/annotation/标注结果-长征非测试集(阳性）/01回流数据信息表（UID)-DD1209V6- 阳（阴）性含测试集数据数量表格.xlsx'
    df = pd.read_excel(anno_file, sheet_name=2)
    
    in_root = os.path.join(data_root, 'Lung_GGO-bk')
    out_root = os.path.join(data_root, 'cz', 'block_{}'.format(block_size),'neg')
    os.makedirs(out_root, exist_ok=True)    

    for index, row in tqdm(df.iterrows()):
        suid = row['series uid']
        image_path = os.path.join(in_root, suid)
        mask_files = glob(os.path.join(image_path, 'mask*'))
        image_file = os.path.join(image_path, 'image.nrrd')
        image = sitk.ReadImage(image_file)
        try: 
            for index, mask_file in enumerate(mask_files):
                z_min, y_min, x_min, z_max, y_max, x_max = MaskBoundingUtils.extract_mask_file_bounding(mask_file)
                pix_coord_center = [int((x_min+x_max)//2), int((y_min+y_max)//2), int((z_min+z_max)//2)]

                image_block = DetectionAnnotationAnalyzeUtils.extract_block_around_center_image(image, pix_coord_center, block_size)

                
                idx = '{}_{}'.format(suid, index+1)
                out_sub_dir = os.path.join(out_root, idx)
                os.makedirs(out_sub_dir, exist_ok=True)
                out_file = os.path.join(out_sub_dir, 'block.nii.gz'.format(block_size))
                sitk.WriteImage(image_block, out_file)
        except Exception as e:
            print('====> Error case:\t', row['series uid'])
            print(e)

def split_ds_cz():
    anno_file = '/fileser/zhangwd/data/hospital/cz/ggo/annotation/group_table/all_datasetByPatient20210408.csv'
    block_root = '/fileser/zhangwd/data/hospital/cz/ggo/cz/block_66'
    config_root = '/fileser/zhangwd/data/hospital/cz/ggo/cz/config'
    os.makedirs(config_root, exist_ok=True)
    df = pd.read_csv(anno_file)
    train_list = []
    val_list = []
    test_list = []
    filenames = glob(os.path.join(block_root, '*/*/*'))
    suids = [i.split('/')[-2].split('_')[0] for i in filenames]
    for index, row in tqdm(df.iterrows()):
        suid = row['series uid']
        mode = row['dataset']
        flag = row['malignancy']
        
        for filename in filenames:
            if suid in filename:
                flag = filename.split('/')[-3]
                if mode == 'train':
                    train_list.append('{}\t{}'.format(filename.replace(block_root+'/', ''), flag))
                elif mode == 'val':
                    val_list.append('{}\t{}'.format(filename.replace(block_root+'/', ''), flag))
                else:
                    test_list.append('{}\t{}'.format(filename.replace(block_root+'/', ''), flag))

    with open(os.path.join(config_root, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_list))
    with open(os.path.join(config_root, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_list))
    with open(os.path.join(config_root, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_list))

    print('hello world!')

if __name__ == '__main__':
    # extract_block_cz_pos(66)
    # extract_block_cz_neg(66)
    # split_ds_cz()
    # extract_block_cz_pos(132)
    # extract_block_cz_neg(132)
    # extract_block_cz_pos(46)
    # extract_block_cz_neg(46)    
    extract_block_cz_pos(270)
    extract_block_cz_neg(270)   