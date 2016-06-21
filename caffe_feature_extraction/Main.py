# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import numpy as np
import sys
caffe_root = '/home/zexe/caffe/'
ffmpeg_root ='/home/zexe/disk1/Downloads/ffmpeg_video/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/home/zexe/caffe/python')
import caffe
import matplotlib.pyplot as plt
import os
import Feature_Extraction as fe


def main():
    model = 'alexnet_p_c_3'
    duration = '2secs/'
    fe.extract_features('fv_' + model + '_' + duration + '.csv', model, duration)


if __name__ == '__main__':
    main()
