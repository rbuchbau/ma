# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe
import FileIO


def main():
    model = 'alexnet_p_c_3'
    # duration = '2secs'
    durations = ['05secs', '1secs', '2secs', '3secs', '4secs', '5secs']
    feature = 'fc6'
    kernels = ['rbf', 'linear']
    # kernels = ['rbf', 'linear', 'poly']
    # filename = '' + model + '_' + duration + '_' + feature
    filename = '' + model + '_' + feature
    # svm_suffix = '_25000'
    svm_s = [10000, 25000, 50000]
    # svm_s = [25000, 50000]
    # svm_s = [10000]
    videos = ['1001', '1004', '1005', '1007', '1009', '1012', '1016', '1017', '1049', '1059']
    # videos = ['1005', '1059']

    #convert model mean from .binaryproto to .npy (needs only be done once for each model
    # Fe.convert_binaryproto_to_npy(model)

    #convert groundtruth from 0.5secs intervall (manual input) to usable intervals (1,2,3,4,5secs)
    # FileIO.convert_groundtruths(videos)

    # #train svm
    for kernel in kernels:
        for svm_size in svm_s:
            svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' \
                       + str(svm_size) + '/svm_' + kernel + '_' + filename + '_' + str(svm_size) + '.pkl'

            #train svm
            Fe.train_and_save_svm(svm_path, model, feature, kernel, svm_size, True)

    # use svm
    # for kernel in kernels:
    #     for duration in durations:
    #         for svm_size in svm_s:
    #             acc = []
    #             info = feature + '_' + kernel + '_' + duration + '_' + str(svm_size)
    #             svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' \
    #                        + str(svm_size) + '/svm_' + kernel + '_' + filename + '_' + str(svm_size) + '.pkl'
    #             for v in videos:
    #                 use svm
                    # acc_values = Fe.load_and_use_svm(svm_path, model, duration, feature, v, svm_size, False)
                    # FileIO.write_accuracy('acc_results/' + info + '.csv', acc_values, info)
                    # acc.append(acc_values[:])
                # avg_acc_values = Fe.calc_average_accuracy(acc)
                # FileIO.write_accuracy('acc_results_avgs/results.csv', avg_acc_values, info)


if __name__ == '__main__':
    main()

