import numpy as np

from utils import CTScan

path = f'''./sample/1.3.6.1.4.1.14519.5.2.1.6279.6001.272042302501586336192628818865.mhd'''
ct = CTScan(path=path)
ct.preprocess()
img = ct.get_image()
np.save(file=f'''./sample/img.npy''', arr=img)
