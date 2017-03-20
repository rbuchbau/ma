import os
import shutil
import errno

path = '../videodataset/'

def createFolders(videofiles, videos):
    for v in videofiles:
        if v.filename in videos:
            try:
                os.makedirs(path + v.id)
            except OSError:
                if not os.path.isdir(path + v.id):
                    raise


def moveFiles(videolist, videos):
    double_elements = []

    for v in videolist:
        if v.filename in videos:
            if os.path.isfile(path + v.filename):
                os.rename(path + v.filename, path + v.id + '/' + v.id + '.mp4')
            else:
                print path + v.filename + ' ' + v.id
                double_elements.append(v)
                # for v2 in videolist:
                #     if v.filename == v2.filename:
                #         shutil.copy2('videodataset/' + v2.id + '/' + v2.id + '.mp4',
                #                      'videodataset/' + v.id + '/' + v.id + '.mp4')

    # with open('../groundtruth/double_elements.txt', 'w') as f:
    #     for d in double_elements:
    #         f.write(d.id + '\n')
    #
    #     f.close()

