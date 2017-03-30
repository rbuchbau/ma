# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe
import FileIO
from concept_crawling import FileIO as ccFileIO


def main():
    models = ['alexnet_p', 'alexnet_p_c', 'alexnet_p_without_weights', 'alexnet_p_c_without_weights']
    features = ['fc6', 'fc7']
    # models = ['alexnet_p_without_weights', 'alexnet_p_c_without_weights']
    # models = ['alexnet_p']

    # svm(models)

    # NEW
    # model(models)

    for model in models:
        #for cnn models
        # FileIO.write_average_accuracies('acc_results_avgs/acc_results.txt', model)

        #for svms
        for feature in features:
            FileIO.write_average_accuracies('acc_results_avgs/acc_results.txt', model + '_' + feature)


def svm(models, features):
    kernel = 'rbf'

    #convert model mean from .binaryproto to .npy (needs only be done once for each model
    # Fe.convert_binaryproto_to_npy(model)

    #train svm
    # train_svm(models, features, kernel)

    # use svm
    use_svm(models, features)


def model(models):
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

    for model in models:
        # convert model mean from .binaryproto to .npy (needs only be done once for each model
        # Fe.convert_binaryproto_to_npy(m)

        acc_values_for_all_concepts = Fe.load_and_use_model(model, all_infos, mapp)
        FileIO.write_accuracies(acc_values_for_all_concepts, model)



def train_svm(models, features, kernel):
    for model in models:
        for feature in features:
            filename = '' + model + '_' + feature
            # svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' \
            #            + str(svm_size) + '/svm_' + kernel + '_' + filename + '_' + str(svm_size) + '.pkl'

            svm_path = 'svms/' + filename + '.pkl'

            #train svm
            Fe.train_and_save_svm(svm_path, model, feature, kernel, True)


# def use_svm(model, features, kernel):
def use_svm(models, features):
    # for feature in features:
    #     acc = []
    #     # info = feature + '_' + kernel + '_' + duration + '_' + str(svm_size)
    #     # svm_path = 'svms/' + feature + '/' + kernel + '/svm_' + kernel + '_' + filename + '_' \
    #     #            + str(svm_size) + '/svm_' + kernel + '_' + filename + '_' + str(svm_size) + '.pkl'
    #
    #     filename = '' + model + '_' + feature
    #     svm_path = 'svms/' + filename + '.pkl'
    #     # for v in videos:
    #         # use svm
    #     acc_values = Fe.load_and_use_svm(svm_path, model, feature, False)
    #     FileIO.write_accuracy('acc_results/' + filename + '_svm.csv', acc_values)
    #     acc.append(acc_values[:])
    #     avg_acc_values = Fe.calc_average_accuracy(acc)
    #     FileIO.write_accuracy('acc_results_avgs/results.csv', avg_acc_values, info)

    groundtruth_path = '../groundtruth/'

    conceptsList = ccFileIO.readConceptTxt(groundtruth_path + 'concepts.txt')
    conceptsList_all = ccFileIO.readConceptTxt(groundtruth_path + 'concepts_all.txt')
    videofiles = ccFileIO.read_videofiles(groundtruth_path + 'needed_videos.txt')
    needed_shots = ccFileIO.read_selected_shots_from_file(groundtruth_path + 'shots.csv', conceptsList_all)
    shot_paths = ccFileIO.read_shot_paths(groundtruth_path + 'shot_paths.txt')

    acc_values = []
    all_infos = (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths)

    mapp = {'75': '1267', '13': '1015', '24': '1261', '21': '1031', '64': '1010', '0': '1006'}

    for model in models:
        for feature in features:
            acc_values_for_all_concepts = Fe.load_and_use_svm(model, feature, all_infos, mapp, True)
            FileIO.write_accuracies(acc_values_for_all_concepts, model + '_' + feature)


# def load_and_use_model_svm(model, feature, all_infos, mapp):
    # acc_values = []
    # feat_vectors = []
    # (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths) = all_infos
    #
    # #init caffe net
    # net, transformer = Fe.init_caffe_net(model)
    # #read images to classify from folder
    # print "Preparing images for model " + model
    # all_images = Fe.load_images_to_classify(transformer, shot_paths)
    #
    # # perform classification
    # print "Classifying for model " + model
    # predicted_labels = Fe.classify_for_svm(net, all_images, feature, feat_vectors)
    #
    # print "Calculating evaluation parameters for model " + model
    # acc_values_for_all_concepts = Fe.calc_accuracy(predicted_labels, model, conceptsList, mapp)
    # FileIO.write_accuracies(acc_values_for_all_concepts, model)
    #
    # return acc_values



if __name__ == '__main__':
    main()

