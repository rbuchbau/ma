# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as fe


def main():
    model = 'alexnet_p_c_3'
    duration = '2secs'
    fe.extract_features('fv_' + model + '_' + duration + '.csv', model, duration)


if __name__ == '__main__':
    main()
