# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as fe


def main():
    model = 'alexnet_p_c_3'
    duration = '2secs'
    feature = 'fc7'
    fe.extract_features('fv_' + model + '_' + duration + '_' + feature + '.csv',
                        model, duration, feature, 2)


if __name__ == '__main__':
    main()
