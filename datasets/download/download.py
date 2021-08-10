import os
import sys
import pandas as pd
import numpy as np

from glob import glob

import fire

root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.path.pardir, os.path.pardir
    )
)

print(root)

sys.path.append(root)

DATA_ROOT = '/data/medical/hospital/cz/ggo'

from external_lib.MedCommon.utils.download_utils import DownloadUtils

# 1. download cz pos image series
download_pth = os.path.join(DATA_ROOT, 'cz/raw/pos/images')
config_file = os.path.join(DATA_ROOT, 'annotation/series_table/cz/pos/文件内网地址信息-导出结果.csv')
DownloadUtils.download_dcms_with_website_multiprocess(download_pth, config_file)




