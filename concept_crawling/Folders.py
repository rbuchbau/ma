import os
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
    for v in videolist:
        if v.filename in videos:
            os.rename('videodataset/' + v.filename, 'videodataset/' + v.id + '/' + v.id + '.mp4')
