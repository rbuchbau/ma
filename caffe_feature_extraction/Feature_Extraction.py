import numpy as np
import sys

from networkx.algorithms.shortest_paths.unweighted import predecessor

import FileIO
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

caffe_root = '/home/zexe/caffe/'
# ffmpeg_root ='/home/zexe/disk1/Downloads/ffmpeg_video/'
ffmpeg_root = '/home/zexe/PycharmProjects/ma/videodataset/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/home/zexe/caffe/python')
import caffe
import matplotlib.pyplot as plt
import os
import scipy.linalg as lan
from sklearn.decomposition import PCA
import timeit
import math
import ConceptMeasurements


def init_caffe_net(model):
    # set display defaults
    # plt.rcparams['figure.figsize'] = (10, 10)  # large images
    # plt.rcparams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    # plt.rcparams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

    model_def = caffe_root + 'new2/models/new/' + model + '/deploy.prototxt'
    model_weights = caffe_root + 'new2/models/new/' + model + '/' + model + '.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(
        caffe_root + 'new2/models/new/' + model + '/mean.npy')  # average over pixels to obtain the mean (BGR) pixel values
    mu = mu.mean(1).mean(1)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(1,  # batch size
                              3,  # 3-channel (BGR) images
                              227, 227)  # image size is 227x227

    return net, transformer


def train_and_save_svm(svm_path, model, feature, kernel, save):
    # init caffe net
    net, transformer = init_caffe_net(model)
    filename = model + '_' + feature

    # feat_vectors = []

    if save:
        # either compute feature vectors and write them
        # labels = read_and_classify_imagenet_images(net, transformer, feature, feat_vectors, svm_size)
        labels, feat_vectors = read_and_classify_imagenet_images(net, transformer, model, feature)
        # write feature vectors
        print "Writing feature vectors to file."
        np.savetxt('feature_vectors_training/' + filename + '_labels.csv', labels, fmt='%i')
        np.savetxt('feature_vectors_training/' + filename + '.csv', feat_vectors, delimiter=',', fmt='%4.4f')
    else:
        # or load them from file
        # maybe not working correct!
        print "Loading feature vectors."
        labels = np.genfromtxt('feature_vectors_training/' + filename + '_labels.csv')
        feat_vectors = np.genfromtxt('feature_vectors_training/' + filename + '.csv',
                                     dtype='float32', delimiter=',')

    if kernel == 'rbf':
        filename = '' + model + '_' + feature
    else:
        filename = '' + model + '_' + feature + '_' + kernel

    print filename

    # svm stuff
    # prepare data for svm training
    X_train = np.array(feat_vectors)
    y_train = np.array(labels)

    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train_scaled = std_scaler.transform(X_train)
    scalername = 'svms/scaler/' + filename + '/' + filename + '.pkl'
    print "Scaler: " + scalername
    joblib.dump(std_scaler, scalername)
    # X_train_scaled = X_train

    # create classifier
    print "Training SVM: " + svm_path
    clf = SVC(kernel=kernel, max_iter=1000, tol=1e-6)
    # clf = SVC(kernel=kernel, max_iter=1000, tol=1e-9)

    # train svm
    s_time = timeit.default_timer()
    clf.fit(X_train_scaled, y_train)
    # clf.fit(X_pca, y_train)

    time = format(timeit.default_timer() - s_time, '.4f')
    print "Time for training: " + str(time) + " seconds."
    FileIO.write_times('training_times/svm_training_times.csv', time, model, feature, kernel)

    # save svm to file
    joblib.dump(clf, svm_path)


def load_and_use_svm(model, feature, all_infos, mapp, kernel, save=False):
    (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths) = all_infos
    filename = model + '_' + feature

    if save:
        # init caffe net
        net, transformer = init_caffe_net(model)

        # read images to classify from folder
        print "Preparing images for model " + model
        all_images = read_images_for_predicting(transformer, shot_paths)

        ### perform classification
        print "Classifying for model " + model
        feat_vectors = []
        classify_for_svm_predicting(net, all_images, feature, feat_vectors)

        # write feature vectors
        print "Writing feature vectors to file: " + filename + "."
        np.savetxt('feature_vectors/' + filename + '.csv', feat_vectors, delimiter=',', fmt='%4.4f')
    else:
        # or load them from file
        # maybe not working correct!
        print "Loading feature vectors from " + filename
        feat_vectors = np.genfromtxt('feature_vectors/' + filename + '.csv',
                                     dtype='float32', delimiter=',')


    if kernel == 'rbf':
        filename = '' + model + '_' + feature
    else:
        filename = '' + model + '_' + feature + '_' + kernel

    # load svm from file
    print "Load SVM: " + filename
    svm = joblib.load('svms/' + filename + '.pkl')

    # prepare data for svm training
    X_test = np.array(feat_vectors)
    std_scaler = joblib.load('svms/scaler/' + filename + '/' + filename + '.pkl')
    X_test_scaled = std_scaler.transform(X_test)
    # X_test_scaled = X_test

    # try pca
    # pca = joblib.load('svms/pca/1.pkl')
    # X_pca = pca.transform(X_test_scaled)

    # let it predict
    print "Start predicting."
    s_time = timeit.default_timer()
    predicted_labels = svm.predict(X_test_scaled)
    # predicted_labels = svm.predict(X_pca)
    time = format(timeit.default_timer() - s_time, '.4f')
    FileIO.write_times('training_times/prediction_times.csv', time, model, feature, kernel)

    print "Calculating evaluation parameters."
    # labels is a numpy array
    pred_labels = []
    predicted_labels_list = predicted_labels.tolist()
    for i, prediction in enumerate(predicted_labels_list):
        pred_labels.append((shot_paths[i].name, str(prediction)))

    acc_values_for_all_concepts = calc_accuracy(pred_labels, conceptsList, mapp)

    return acc_values_for_all_concepts


def load_and_use_model(model, all_infos, mapp):
    (conceptsList, conceptsList_all, videofiles, needed_shots, shot_paths) = all_infos

    # init caffe net
    net, transformer = init_caffe_net(model)
    # read images to classify from folder
    print "Preparing images for model " + model
    all_images = read_images_for_predicting(transformer, shot_paths)

    # perform classification
    print "Classifying for model " + model
    s_time = timeit.default_timer()
    predicted_labels = classify(net, all_images)
    time = format(timeit.default_timer() - s_time, '.4f')
    FileIO.write_times('training_times/classification_times.csv', time, model, '', len(predicted_labels))

    print "Calculating evaluation parameters for model " + model
    acc_values_for_all_concepts = calc_accuracy(predicted_labels, conceptsList, mapp)

    return acc_values_for_all_concepts


def convert_binaryproto_to_npy(model):
    # convert .binaryproto to .npy
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(caffe_root + 'new2/models/new/' + model + '/mean.binaryproto', 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    out = arr[0]
    np.save(caffe_root + 'new2/models/new/' + model + '/mean.npy', out)


def classify(net, all_images, index=0):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    labels = []

    for i, (img, shot_path) in enumerate(all_images):
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = img

        output = net.forward()

        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

        labels.append((shot_path.name, str(output_prob.argmax())))
        # labels.append(str(output_prob.argmax()))

        if i % 1000 == 0:
            print "Classified " + str(index + i) + " images."

    return labels


def classify_for_svm_training(net, all_images, feature, feat_vectors, index=0):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # for i, (img, shot_path) in enumerate(all_images):
    for i, img in enumerate(all_images):
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = img

        output = net.forward()

        # get features of one layer
        feat = net.blobs[feature].data[0]
        feat = feat.flat
        feat_vectors.append(np.array(feat[:]))

        if i % 1000 == 0:
            print "Classified " + str(index + i) + " images."

def classify_for_svm_predicting(net, all_images, feature, feat_vectors, index=0):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    for i, (img, shot_path) in enumerate(all_images):
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = img

        output = net.forward()

        # get features of one layer
        feat = net.blobs[feature].data[0]
        feat = feat.flat
        feat_vectors.append(np.array(feat[:]))

        if i % 1000 == 0:
            print "Classified " + str(index + i) + " images."


def read_and_classify_imagenet_images(net, transformer, model, feature):
    svm_size = 25000
    # read images and labels from disk
    print "Preparing and classifying images."
    # read text file with labels
    splits = model.split('_without')
    image_tuples = FileIO.read_csv(
        'paths/' + splits[0] + '_train_svm.txt')  # returns list of tupels (file_path, label)

    length = 1000
    labels = []

    feat_vectors = []

    for offset in range(0 / length, svm_size / length):
        all_images = read_images_for_training(transformer, image_tuples, labels, offset * length, length)
        classify_for_svm_training(net, all_images, feature, feat_vectors, offset * length)

    return labels, feat_vectors


def read_images_for_training(transformer, image_tuples, labels, offset, length):
    all_images = []
    file_paths = []

    # labels = []
    error_number = 0

    for i, (fp, label) in enumerate(image_tuples):
        if offset <= i < (offset + length):
            try:
                image = caffe.io.load_image('/home/zexe/' + fp)
                transformed_image = transformer.preprocess('data', image)
                all_images.append(transformed_image)
                labels.append(label)
            except:
                print "Error reading image (Probably not an image). # " + str(error_number)
                error_number += 1

            if i % 1000 == 0:
                print "Read " + str(i) + " images."

            if i == offset + length - 1:
                break

    return all_images


def read_images_for_predicting(transformer, shot_paths):
    all_images = []

    # get number of files in directory
    path = ffmpeg_root
    num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))]) - 1

    # load and preprocess all image files
    for i, shot_path in enumerate(shot_paths):
        # if i > 20746:
        image = caffe.io.load_image(path + shot_path.path)
        transformed_image = transformer.preprocess('data', image)

        all_images.append((transformed_image, shot_path))

        if i % 1000 == 0:
            print "Read " + str(i) + " images."
            # if i == 23000:
            #     break

    print "Read all images."

    return all_images


def calc_accuracy(predicted_labels, conceptsList, mapp):
    # dictionary of all concepts:
    #   keys = conceptIDs,
    #   values = true_positives, relevant_images, corpus_images, precision, recall, f_measure
    all_concepts = {}
    for concept in conceptsList.dictionary.keys():
        all_concepts[concept] = ConceptMeasurements.ConceptMeasurements()
        all_concepts[concept].id = concept
        all_concepts[concept].list_of_true_positives = []
        all_concepts[concept].list_of_all_detected_images = []

        # get all relevant corpus images
        all_concepts[concept].list_of_all_corpus_images = conceptsList.dictionary[concept].shots[:]

    # set concept id
    for k in mapp.keys():
        for concept in all_concepts.values():
            if mapp[k] == concept.id:
                concept.name = k

    # work the labels
    # for shot, label in predicted_labels:
    for i, (shot, label) in enumerate(predicted_labels):
        if label in mapp.keys():
            # => is a relevant concept
            conceptID = mapp[label]
            concept = conceptsList.dictionary[conceptID]
            # get true positives
            if shot in concept.shots:
                all_concepts[concept.name].list_of_true_positives.append(shot[:])
            # get all detected images
            all_concepts[concept.name].list_of_all_detected_images.append(shot[:])

    # # work the labels (binary models variant)
    # # for shot, label in predicted_labels:
    # for i, (shot, label) in enumerate(predicted_labels):
    #     if label in mapp.keys():
    #         # => is a relevant concept
    #         if label == '0':
    #             conceptID = mapp[label]
    #             concept = conceptsList.dictionary[conceptID]
    #             # get true positives
    #             if shot in concept.shots:
    #                 all_concepts[concept.name].list_of_true_positives.append(shot[:])
    #             # get all detected images
    #             all_concepts[concept.name].list_of_all_detected_images.append(shot[:])

    for concept in all_concepts.values():
        concept.calc_precision()
        concept.calc_recall()
        concept.calc_f_measure()

    return all_concepts

