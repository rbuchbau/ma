# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe
import FileIO
from concept_crawling import FileIO as ccFileIO


def main():
    models = ['alexnet_p', 'alexnet_p_c', 'alexnet_p_without_weights', 'alexnet_p_c_without_weights']

    svm()

    # NEW
    # model(models)
    FileIO.read_accuracies_and_average_them('acc_results/', models)


def svm():
    models = ['alexnet_p']
    # duration = '2secs'
    # durations = ['05secs', '1secs', '2secs', '3secs', '4secs', '5secs']
    feature = 'fc7'
    kernels = ['rbf']
    # kernels = ['rbf', 'linear', 'poly']
    # filename = '' + model + '_' + duration + '_' + feature
    # svm_suffix = '_25000'
    svm_s = [25000]
    # svm_s = [25000, 50000]
    # svm_s = [10000]
    # videos = ['1001', '1004', '1005', '1007', '1009', '1012', '1016', '1017', '1049', '1059']
    # videos = ['1005', '1059']

    #convert model mean from .binaryproto to .npy (needs only be done once for each model
    # Fe.convert_binaryproto_to_npy(model)

    #convert groundtruth from 0.5secs intervall (manual input) to usable intervals (1,2,3,4,5secs)
    # FileIO.convert_groundtruths(videos)

    # #train svm
    for model in models:
        filename = '' + model + '_' + feature
        for kernel in kernels:
            for svm_size in svm_s:
                # svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' \
                #            + str(svm_size) + '/svm_' + kernel + '_' + filename + '_' + str(svm_size) + '.pkl'

                svm_path = 'svms/' + filename + '.pkl'

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


def model():
    groundtruth_path = '../groundtruth/'
    # models = ['alexnet_p']
    # or read them from csv
    conceptsList = ccFileIO.readConceptTxt(groundtruth_path + 'concepts.txt')
    conceptsList_all = ccFileIO.readConceptTxt(groundtruth_path + 'concepts_all.txt')
    videofiles = ccFileIO.read_videofiles(groundtruth_path + 'needed_videos.txt')
    needed_shots = ccFileIO.read_selected_shots_from_file(groundtruth_path + 'shots.csv', conceptsList_all)
    shot_paths = ccFileIO.read_shot_paths(groundtruth_path + 'shot_paths.txt')

    all_infos = (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths)

    # 1267 forest <-> 75 tree
    # 1015 boat / ship <-> 13 boat
    # 1261 flags <-> 24 flag
    # 1031 computers <-> 21 computer
    # 1010 beach <-> 64 beach
    # 1006 animal <-> 0 animal

    mapp = {'75': '1267', '13': '1015', '24': '1261', '21': '1031', '64': '1010', '0': '1006'}

    for m in models:
        # convert model mean from .binaryproto to .npy (needs only be done once for each model
        Fe.convert_binaryproto_to_npy(m)

        acc_values = load_and_use_model(m, all_infos, mapp)


def load_and_use_model(model, all_infos, mapp):
    acc_values = []
    (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths) = all_infos

    #init caffe net
    net, transformer = Fe.init_caffe_net(model)
    #read images to classify from folder
    print "Preparing images for model " + model
    all_images = Fe.load_images_to_classify(transformer, shot_paths)

    # perform classification
    print "Classifying for model " + model
    predicted_labels = Fe.classify(net, all_images, model, conceptsList, mapp, needed_shots)

    print "Calculating evaluation parameters for model " + model
    acc_values_for_all_concepts = Fe.calc_accuracy(predicted_labels, model, conceptsList, mapp)
    FileIO.write_accuracies(acc_values_for_all_concepts, model)

    return acc_values






if __name__ == '__main__':
    main()

