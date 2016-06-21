def extract_features(filename, model, duration):
    # set display defaults
    plt.rcparams['figure.figsize'] = (10, 10)  # large images
    plt.rcparams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcparams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

    model_def = caffe_root + 'new2/models/' + model + '/deploy.prototxt'
    model_weights = caffe_root + 'new2/models/' + model + '/' + model + '.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # convert .binaryproto to .npy
    # blob = caffe.proto.caffe_pb2.BlobProto()
    # data = open( caffe_root + 'new2/models/alexnet_p_c_3/mean.binaryproto', 'rb' ).read()
    # blob.ParseFromString(data)
    # arr = np.array( caffe.io.blobproto_to_array(blob) )
    # out = arr[0]
    # np.save( caffe_root + 'new2/models/alexnet_p_c_3/mean.npy' , out )

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

    all_images = []

    # get number of files in directory
    path = ffmpeg_root + duration
    num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))]) - 1

    # load and preprocess all image files
    for i in range(1, num_files):
        image = caffe.io.load_image(ffmpeg_root + duration + str(i) + '.jpg')
        transformed_image = transformer.preprocess('data', image)
        all_images.append(transformed_image)
    # plt.imshow(image)
    # plt.show()

    ### perform classification
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
        feat = net.blobs['fc7'].data[0]
        feat = feat.flat
        feat_vectors.append(feat[:])

    print str(len(feat_vectors))
    with open('feature_vectors/' + filename, 'w') as f:
        for i in range(0, len(feat_vectors)):
            for j in range(0, len(feat_vectors[i]) - 1):
                value = feat_vectors[i][j]
                if abs(value) < 0.01:
                    value = 0
                value = format(value, '.4f')
                f.write(str(value) + ',')

            f.write(str(value) + '\n')

        f.close()


        # print "Plotting."
        # plt.plot(feat)
        # # plt.show()