# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import Feature_Extraction as Fe
import FileIO
from concept_crawling import FileIO as ccFileIO


def main():
    models = ['alexnet_p', 'alexnet_p_c', 'alexnet_p_without_weights', 'alexnet_p_c_without_weights']
    models_short = ['alexnet_p', 'alexnet_p_ww', 'alexnet_p_c', 'alexnet_p_c_ww']
    # models = ['alexnet']
    # features = ['fc6', 'fc7']
    features = ['fc7']
    # kernels = ['linear', 'rbf']
    kernels = ['rbf']
    features_out = ['', '_fc6', '_fc7', '_fc6_linear', '_fc7_linear']
    mapp = {'75': '1267', '13': '1015', '24': '1261', '21': '1031', '64': '1010', '0': '1006'}
    # mapp = {'0': '1267'}   # for binary models, change depending on the used conceptID

    # svm(models, features, mapp, kernels)

    # model(models, mapp)

    formatAccuracies(models, features_out, models_short)

    # calc average accuracies
    # calcAverageAccuracies(models, features)

    # # modify train.txt and val.txt for usage with binary models, can be commented after single use
    # modifyTrainTxt()
    # # modify synset_words.txt for usage with binary models
    # modifySynsetWordsTxt()


def svm(models, features, mapp, kernels):

    #convert model mean from .binaryproto to .npy (needs only be done once for each model
    # for model in models:
    #     Fe.convert_binaryproto_to_npy(model)

    #train svm
    for kernel in kernels:
        train_svm(models, features, kernel)
        # use svm
        use_svm(models, features, mapp, kernel)



def model(models, mapp):
    groundtruth_path = '../groundtruth/'
    # read them from csv
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

    for model in models:
        # convert model mean from .binaryproto to .npy (needs only be done once for each model
        # Fe.convert_binaryproto_to_npy(m)

        acc_values_for_all_concepts = Fe.load_and_use_model(model, all_infos, mapp)
        FileIO.write_accuracies(acc_values_for_all_concepts, model)



def train_svm(models, features, kernel):
    for model in models:
        for feature in features:
            if kernel == 'rbf':
                filename = '' + model + '_' + feature
            else:
                filename = '' + model + '_' + feature + '_' + kernel

            svm_path = 'svms/' + filename + '.pkl'

            #train svm
            Fe.train_and_save_svm(svm_path, model, feature, kernel, True)


# def use_svm(model, features, kernel):
def use_svm(models, features, mapp, kernel):

    groundtruth_path = '../groundtruth/'

    conceptsList = ccFileIO.readConceptTxt(groundtruth_path + 'concepts.txt')
    conceptsList_all = ccFileIO.readConceptTxt(groundtruth_path + 'concepts_all.txt')
    videofiles = ccFileIO.read_videofiles(groundtruth_path + 'needed_videos.txt')
    needed_shots = ccFileIO.read_selected_shots_from_file(groundtruth_path + 'shots.csv', conceptsList_all)
    shot_paths = ccFileIO.read_shot_paths(groundtruth_path + 'shot_paths.txt')

    acc_values = []
    all_infos = (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths)

    for model in models:
        for feature in features:
            acc_values_for_all_concepts = Fe.load_and_use_svm(model, feature, all_infos, mapp, kernel, True)

            if kernel == 'rbf':
                filename = '' + model + '_' + feature
            else:
                filename = '' + model + '_' + feature + '_' + kernel
            FileIO.write_accuracies(acc_values_for_all_concepts, filename)


def formatAccuracies(models, features_out, models_short):
    # format output accuracies
    for i, model in enumerate(models):
        model_short = models_short[i]
        for feature in features_out:
            FileIO.format_accuracies('acc_results/' + model + feature + '.txt',
                                     'acc_results_formatted/' + model + feature + '.txt',
                                     'acc_results_per_concept/', model_short + feature)


def calcAverageAccuracies(models, features):

    #header lines
    with open('acc_results_avgs/acc_results.txt', 'w') as f:
        f.write('Model_Feature Precis. Recall F-Meas.\n')
        f.close()
    with open('acc_results_avgs/acc_results_only_svms.txt', 'w') as f:
        f.write('Feature Precis. Recall F-Meas.\n')
        f.close()
    with open('acc_results_avgs/acc_results_only_models.txt', 'w') as f:
        f.write('Model Precis. Recall F-Meas.\n')
        f.close()

    #for cnn models
    for model in models:
        FileIO.write_average_accuracies('acc_results_avgs/acc_results_only_models.txt', model)
        #for svms
        FileIO.write_average_accuracies('acc_results_avgs/acc_results.txt', model)
        for feature in features:
            FileIO.write_average_accuracies('acc_results_avgs/acc_results.txt', model + '_' + feature)
            FileIO.write_average_accuracies('acc_results_avgs/acc_results_only_svms.txt', model + '_' + feature)


def modifyTrainTxt():
    mapp = {'75': '1267', '13': '1015', '24': '1261', '21': '1031', '64': '1010', '0': '1006'}
    data_types = ['data_p', 'data_p_c']
    for id in mapp:
        for dt in data_types:
            FileIO.modify_train('modified_training_files/' + dt + '/train.txt',
                                'modified_training_files/' + dt + '/train_' + id + '.txt', id)
            FileIO.modify_train('modified_training_files/' + dt + '/val.txt',
                                'modified_training_files/' + dt + '/val_' + id + '.txt', id)


def modifySynsetWordsTxt():
    synset_words = {'0': 'animal', '13': 'boat', '21': 'computer', '24': 'flag', '64': 'beach', '75': 'tree'}
    for id in synset_words:
        FileIO.modify_synset_words('synset_words.txt', 'modified_training_files/synset_words_' + id + '.txt', id,
                                   synset_words[id])


if __name__ == '__main__':
    main()
