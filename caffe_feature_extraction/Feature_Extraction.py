
import numpy as np
import sys
import FileIO
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
caffe_root = '/home/zexe/caffe/'
ffmpeg_root ='/home/zexe/disk1/Downloads/ffmpeg_video/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/home/zexe/caffe/python')
import caffe
import matplotlib.pyplot as plt
import os
import scipy.linalg as lan
from sklearn.decomposition import PCA


def train_and_save_svm(svm_path, model, feature, kernel):
    #init caffe net
    net, transformer = init_caffe_net(model)

    #read images and labels from disk
    print "Preparing images."
    all_images, labels = read_images_and_labels(transformer)

    ### perform classification
    print "Classifying."
    feat_vectors = classify(net, feature, all_images)

    #svm stuff

    #prepare data for svm training
    X_train = np.array(feat_vectors)
    y_train = np.array(labels)

    std_scaler = StandardScaler()
    # X_train_scaled = std_scaler.fit_transform(X_train)
    std_scaler.fit(X_train)
    X_train_scaled = std_scaler.transform(X_train)
    joblib.dump(std_scaler, 'svms/scaler/1.pkl')
    # X_train_scaled = X_train

    # try pca first
    # print "Fit PCA"
    # pca = PCA(n_components=2)
    # pca.fit(X_train_scaled)
    # X_pca = pca.transform(X_train_scaled)
    # joblib.dump(pca, 'svms/pca/1.pkl')


    #create classifier
    print "Training SVM"
    clf = SVC(kernel=kernel, max_iter=1000, tol=1e-6)

    #train svm
    clf.fit(X_train_scaled, y_train)
    # clf.fit(X_pca, y_train)

    #save svm to file
    joblib.dump(clf, svm_path)


def load_and_use_svm(filename, svm_path, model, duration, feature, save=False):
    feature_vectors = 0
    if save:
        #init caffe net
        net, transformer = init_caffe_net(model)
        #read images to classify from folder
        print "Preparing images."
        all_images = load_images_to_classify(transformer, duration)

        ### perform classification
        print "Classifying."
        feat_vectors = classify(net, feature, all_images)

        #write feature vectors
        print "Writing feature vectors to file."
        np.savetxt('feature_vectors/fv_' + filename + '.csv', feat_vectors, delimiter=',', fmt='%4.4f')
    else:
        #load feature vectors from file
        print "Loading feature vectors."
        feat_vectors = np.genfromtxt('feature_vectors/fv_' + filename + '.csv',  dtype='float32', delimiter=',')

    #load svm from file
    print "Load SVM: " + svm_path
    svm = joblib.load(svm_path)

    #try norm
    # feat_vectors2 = []
    # for f in feat_vectors:
    #     vec = []
    #     norm = lan.norm(f)
    #     vec.append(norm)
    #     vec.append(norm)
    #     feat_vectors2.append(np.array(vec[:]))
    # X_test = np.array(feat_vectors2)

    # try norm second version
    feat_vectors2 = []
    for feat in feat_vectors:
        feat_np = np.array(feat[:])
        norm = lan.norm(feat_np)
        feat_norm = np.divide(feat_np, norm)
        feat_vectors2.append(np.array(feat_norm[:]))
    X_test = np.array(feat_vectors2)

    #prepare data for svm training
    # X_test = np.array(feat_vectors)
    std_scaler = joblib.load('svms/scaler/1.pkl')
    X_test_scaled = std_scaler.transform(X_test)
    # X_test_scaled = X_test

    #try pca
    # pca = joblib.load('svms/pca/1.pkl')
    # X_pca = pca.transform(X_test_scaled)

    #let it predict
    print "Start predicting."
    predicted_labels = svm.predict(X_test_scaled)
    # predicted_labels = svm.predict(X_pca)

    print "Calculating evaluation parameters."
    calc_accuracy(predicted_labels, 'groundtruth_1001_' + duration + '.csv')


def init_caffe_net(model):
    # set display defaults
    # plt.rcparams['figure.figsize'] = (10, 10)  # large images
    # plt.rcparams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    # plt.rcparams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

    model_def = caffe_root + 'new2/models/' + model + '/deploy.prototxt'
    model_weights = caffe_root + 'new2/models/' + model + '/' + model + '.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(
        caffe_root + 'new2/models/' + model + '/mean.npy')  # average over pixels to obtain the mean (BGR) pixel values
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


def convert_binaryproto_to_npy(model):
    # convert .binaryproto to .npy
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( caffe_root + 'new2/models/' + model + '/mean.binaryproto', 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    np.save( caffe_root + 'new2/models/' + model + '/mean.npy' , out )


def classify(net, feature, all_images):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    feat_vectors = []

    for i, img in enumerate(all_images):
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = img

        output = net.forward()

        # output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
        # print 'predicted class for ' + str(i+1) + '.jpg is:', output_prob.argmax()
        # # load ImageNet labels
        # labels_file = caffe_root + 'new2/models/alexnet_p_c_3/synset_words.txt'
        # labels = np.loadtxt(labels_file, str, delimiter='\t')
        # print 'output label:', labels[output_prob.argmax()]

        # get features of one layer
        feat = net.blobs[feature].data[0]
        feat = feat.flat
        # feat_vectors.append(np.array(feat[:]))

        #try norm
        # norm = lan.norm(np.array(feat[:]))
        # vec = []
        # vec.append(norm)
        # vec.append(norm)
        # feat_vectors.append(np.array(vec[:]))

        #try norm second version
        feat_np = np.array(feat[:])
        norm = lan.norm(feat_np)
        feat_norm = np.divide(feat_np, norm)
        feat_vectors.append(np.array(feat_norm[:]))

        if i % 1000 == 0:
            print "Classified " + str(i) + " images."

    return feat_vectors


def read_images_and_labels(transformer):
    all_images = []
    labels = []
    file_paths = []

    #read text file with labels
    data = FileIO.read_csv('/home/zexe/disk1/Downloads/original_data/data_small/val.txt')   #returns list of tupels (file_path, label)
    for fp, label in data:
        file_paths.append(fp)
        # labels.append(label)

    for i, (fp, label) in enumerate(data):
        # file_paths.append(fp)
        labels.append(label)
        # if i == 1000:
        #     break

    for i, fp in enumerate(file_paths):
        image = caffe.io.load_image('/home/zexe/disk1/Downloads/original_data/' + fp)
        transformed_image = transformer.preprocess('data', image)
        all_images.append(transformed_image)

        if i % 1000 == 0:
            print "Read " + str(i) + " images."

        # if i == 1000:
        #     break

    return all_images, labels


def load_images_to_classify(transformer, duration):
    all_images = []

    # get number of files in directory
    path = ffmpeg_root + duration + '/'
    num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))]) - 1

    # load and preprocess all image files
    for i in range(1, num_files):
        image = caffe.io.load_image(ffmpeg_root + duration + '/' + str(i) + '.jpg')
        transformed_image = transformer.preprocess('data', image)
        all_images.append(transformed_image)

    print "Read all images."
    # plt.imshow(image)
    # plt.show()

    return all_images


def get_labels(filename='synset_words.txt'):
    data = FileIO.read_synset_words(filename)

    labels = []

    for i,cls in enumerate(data):
        if cls == 'sport' or cls == 'water sport':
            labels.append(i)

    return labels


def calc_accuracy(predicted_labels, filename_csv):
    #for prec, recall, ... calculation
    ground_truth = FileIO.read_groundtruth(filename_csv)
    labels = get_labels()

    list_of_groundtruth_images = []

    #refactor groundtruth
    for (a,b) in ground_truth:
        for i in range(a,b+1):
            list_of_groundtruth_images.append(i)

    list_of_relevant_images = []

    for i, label in enumerate(predicted_labels):
        if label in labels:
            list_of_relevant_images.append( + 1)

    list_of_true_positives = []

    for ri in list_of_relevant_images:
        if ri in list_of_groundtruth_images:
            list_of_true_positives.append(ri)

    print str(len(list_of_groundtruth_images))
    print str(len(list_of_relevant_images))
    print str(len(list_of_true_positives))

    precision = float(len(list_of_true_positives)) / len(list_of_relevant_images)
    recall = float(len(list_of_true_positives)) / len(list_of_groundtruth_images)
    f_measure = 2 * float(precision*recall) / (precision+recall)

    accuracy = float(
        len(list_of_true_positives) + (
            904 - len(list_of_groundtruth_images)) -
                     (len(list_of_relevant_images) - len(list_of_true_positives))) / 904

    precision = format(precision, '.4f')
    recall = format(recall, '.4f')
    f_measure = format(f_measure, '.4f')
    accuracy = format(accuracy, '.4f')

    print "Precision: " + str(precision)
    print "Recall: " + str(recall)
    print "F-measure: " + str(f_measure)
    print "Accuracy: " + str(accuracy)

    return float(precision), float(recall), float(f_measure)