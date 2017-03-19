import os
import shutil
import errno

def createFolders(videofiles, videos):
    for v in videofiles:
        if v.filename in videos:
            try:
                os.makedirs('videodataset/' + v.id)
            except OSError:
                if not os.path.isdir('videodataset/' + v.id):
                    raise


def moveFiles(videolist, videos):
    double_elements = []

    for v in videolist:
        if v.filename in videos:
            if os.path.isfile('videodataset/' + v.filename):
                os.rename('videodataset/' + v.filename, 'videodataset/' + v.id + '/' + v.id + '.mp4')
            else:
                print 'videodataset/' + v.filename + ' ' + v.id
                double_elements.append(v)
                # for v2 in videolist:
                #     if v.filename == v2.filename:
                #         shutil.copy2('videodataset/' + v2.id + '/' + v2.id + '.mp4',
                #                      'videodataset/' + v.id + '/' + v.id + '.mp4')

    with open('double_elements.txt', 'w') as f:
        for d in double_elements:
            f.write(d.id + '\n')

        f.close()

