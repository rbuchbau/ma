# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe
import FileIO
import concept_crawling as cc


def main():
    # model = 'alexnet_p_c_3'
    # # duration = '2secs'
    # durations = ['05secs', '1secs', '2secs', '3secs', '4secs', '5secs']
    # feature = 'fc6'
    # kernels = ['rbf', 'linear']
    # # kernels = ['rbf', 'linear', 'poly']
    # # filename = '' + model + '_' + duration + '_' + feature
    # filename = '' + model + '_' + feature
    # # svm_suffix = '_25000'
    # svm_s = [10000, 25000, 50000]
    # # svm_s = [25000, 50000]
    # # svm_s = [10000]
    # videos = ['1001', '1004', '1005', '1007', '1009', '1012', '1016', '1017', '1049', '1059']
    # # videos = ['1005', '1059']
    #
    # #convert model mean from .binaryproto to .npy (needs only be done once for each model
    # # Fe.convert_binaryproto_to_npy(model)
    #
    # #convert groundtruth from 0.5secs intervall (manual input) to usable intervals (1,2,3,4,5secs)
    # # FileIO.convert_groundtruths(videos)
    #
    # # #train svm
    # for kernel in kernels:
    #     for svm_size in svm_s:
    #         svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' \
    #                    + str(svm_size) + '/svm_' + kernel + '_' + filename + '_' + str(svm_size) + '.pkl'
    #
    #         #train svm
    #         Fe.train_and_save_svm(svm_path, model, feature, kernel, svm_size, True)
    #
    # # use svm
    # # for kernel in kernels:
    # #     for duration in durations:
    # #         for svm_size in svm_s:
    # #             acc = []
    # #             info = feature + '_' + kernel + '_' + duration + '_' + str(svm_size)
    # #             svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' \
    # #                        + str(svm_size) + '/svm_' + kernel + '_' + filename + '_' + str(svm_size) + '.pkl'
    # #             for v in videos:
    # #                 use svm
    #                 # acc_values = Fe.load_and_use_svm(svm_path, model, duration, feature, v, svm_size, False)
    #                 # FileIO.write_accuracy('acc_results/' + info + '.csv', acc_values, info)
    #                 # acc.append(acc_values[:])
    #             # avg_acc_values = Fe.calc_average_accuracy(acc)
    #             # FileIO.write_accuracy('acc_results_avgs/results.csv', avg_acc_values, info)




    # NEW

    groundtruth_path = '../groundtruth/'
    # models = ['alexnet_p', 'alexnet_p_c', 'alexnet_p_without_weights', 'alexnet_p_c_without_weights']
    models = ['alexnet_p']
    # or read them from csv
    conceptsList = cc.FileIO.readConceptTxt(groundtruth_path + 'concepts.txt')
    conceptsList_all = cc.FileIO.readConceptTxt(groundtruth_path + 'concepts_all.txt')
    videofiles = cc.FileIO.read_videofiles(groundtruth_path + 'needed_videos.txt')
    needed_shots = cc.FileIO.read_selected_shots_from_file(groundtruth_path + 'shots.csv', conceptsList_all)
    shot_paths = cc.FileIO.read_shot_paths(groundtruth_path + 'shot_paths.txt')

    all_infos = (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths)

    for m in models:

        # convert model mean from .binaryproto to .npy (needs only be done once for each model
        Fe.convert_binaryproto_to_npy(model)

        # convert groundtruth from 0.5secs intervall (manual input) to usable intervals (1,2,3,4,5secs)
        # FileIO.convert_groundtruths(videos)

        acc_values = load_and_use_model(m, all_infos)
        info = m
        FileIO.write_accuracy('../results/acc_results/' + info + '.csv', acc_values, info)


def load_and_use_model(model, all_infos):
    # all_infos: (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths)
    acc_values = []

    #init caffe net
    net, transformer = init_caffe_net(model)
    #read images to classify from folder
    print "Preparing images."
    all_images = Fe.load_images_to_classify(transformer, all_infos[4])

    # perform classification
    print "Classifying."
    feat_vectors = []
    Fe.classify(net, [], all_images, feat_vectors)

    print "Calculating evaluation parameters."
    acc_values = Fe.calc_accuracy(predicted_labels, video + '/groundtruth_' + video + '_' + duration + '.csv', video, all_infos[0])
    (p, r, f) = acc_values
    acc_values2 = (p, r, f, time, len(X_test_scaled))




    return acc_values






if __name__ == '__main__':
    main()

