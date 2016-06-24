# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe


def main():
    model = 'alexnet_p_c_3'
    duration = '2secs'
    feature = 'fc7'
    kernel = 'linear'
    filename = '' + model + '_' + duration + '_' + feature

    # Fe.convert_binaryproto_to_npy(model)
    Fe.get_labels()

    # Fe.extract_features(filename, model, duration, feature)

    # svm_path = 'svms/svm_' + kernel + '_' + filename + '/svm_' + kernel + '_' + filename + '.pkl'
    # Fe.train_and_save_svm(svm_path, model, feature, kernel)
    # Fe.load_and_use_svm(filename, svm_path, model, duration, feature)


if __name__ == '__main__':
    main()
