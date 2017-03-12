# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe
import FileIO


def main():
    model = 'alexnet_p_c_3'
    duration = '2secs'
    feature = 'fc7'
    kernel = 'rbf'
    filename = '' + model + '_' + duration + '_' + feature
    # svm_suffix = '_25000'
    # svm_s = [10000, 25000, 50000, 100000]
    svm_s = [16000]
    videos = ['1001', '1004', '1005', '1007', '1009', '1012', '1016', '1017', '1049', '1059']
    # videos = ['1001']

    #convert model mean from .binaryproto to .npy (needs only be done once for each model
    # Fe.convert_binaryproto_to_npy(model)

    #convert groundtruth from 0.5secs intervall (manual input) to usable intervals (1,2,3,4,5secs)
    # FileIO.convert_groundtruths(videos)

    # for svm_size in svm_s:
    #     svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' \
    #                + str(svm_size) + '_step/svm_' + kernel + '_' + filename + '_' + str(svm_size) + '.pkl'
    #
    #     #train svm
    #     Fe.train_and_save_svm(svm_path, model, feature, kernel, svm_size, True)

    #use svm
    svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' + str(svm_s[0]) + '_step/svm_' \
               + kernel + '_' + filename + '_' + str(svm_s[0]) + '.pkl'
    v = videos[0]
    Fe.load_and_use_svm(filename, svm_path, model, duration, feature, v, svm_s[0], False)


if __name__ == '__main__':
    main()

