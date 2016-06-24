# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe


def main():
    model = 'alexnet_p_c_3'
    duration = '2secs'
    feature = 'fc7'
    filename = '' + model + '_' + duration + '_' + feature

    # Fe.convert_binaryproto_to_npy(model)

    # Fe.extract_features(filename, model, duration, feature)

    svm_path = 'svms/svm_linear_' + filename + '/' + filename + '.pkl'
    # Fe.train_and_save_svm(svm_path, model, feature)
    Fe.load_and_use_svm(svm_path, duration)


if __name__ == '__main__':
    main()
