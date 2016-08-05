# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe


def main():
    model = 'alexnet_p_c_3'
    duration = '2secs'
    feature = 'fc7'
    kernel = 'rbf'
    filename = '' + model + '_' + duration + '_' + feature
    svm_suffix = '_normalized_norm2'

    # Fe.convert_binaryproto_to_npy(model)

    svm_path = 'svms/svm_' + kernel + '_' + filename + svm_suffix + '/svm_' \
               + kernel + '_' + filename + svm_suffix + '.pkl'
    Fe.train_and_save_svm(svm_path, model, feature, kernel)
    Fe.load_and_use_svm(filename, svm_path, model, duration, feature, False)


if __name__ == '__main__':
    main()
