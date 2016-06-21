# this code comes from the python binding tutorial:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import numpy as np
import sys
caffe_root = '../../../caffe/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/home/zexe/caffe/python')
import caffe
import matplotlib.pyplot as plt
import os


def main():

    # set display defaults
    plt.rcParams['figure.figsize'] = (10, 10)        # large images
    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

    model_def = caffe_root + 'new2/models/alexnet_p_c_3/deploy.prototxt'
    model_weights = caffe_root + 'new2/models/alexnet_p_c_3/alexnet_p_c_3.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    #convert .binaryproto to .npy
    # blob = caffe.proto.caffe_pb2.BlobProto()
    # data = open( caffe_root + 'new2/models/alexnet_p_c_3/mean.binaryproto', 'rb' ).read()
    # blob.ParseFromString(data)
    # arr = np.array( caffe.io.blobproto_to_array(blob) )
    # out = arr[0]
    # np.save( caffe_root + 'new2/models/alexnet_p_c_3/mean.npy' , out )

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'new2/models/alexnet_p_c_3/mean.npy')  # average over pixels to obtain the mean (BGR) pixel values
    mu = mu.mean(1).mean(1)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

    image = caffe.io.load_image(caffe_root + '../disk1/Downloads/ffmpeg_video/2secs/1.jpg')
    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image)
    #plt.show()


    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    caffe.set_device(0)
    caffe.set_mode_gpu()

    for i in range(0, 100):
        output = net.forward()

        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
        print 'predicted class is:', output_prob.argmax()
        # load ImageNet labels
        labels_file = caffe_root + 'new2/models/alexnet_p_c_3/synset_words.txt'
        labels = np.loadtxt(labels_file, str, delimiter='\t')
        print 'output label:', labels[output_prob.argmax()]


    feat = net.blobs['fc7'].data[0]
    feat = feat.flat

    print "Plotting."
    plt.plot(feat)
    # plt.show()
    print feat[0]


if __name__ == '__main__':
    main()