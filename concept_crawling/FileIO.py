import csv
import xml.etree.ElementTree as ET

import Concept
import ConceptsList
import VideoFile
import os
import Shot


# read ground-truth
def read_labels(filename, conceptsList, double_videos):
    with open(filename, 'r') as f:  # open file for reading
        conceptid = ''
        conceptshot = ''
        conceptgroundtruth = ''
        alllines = f.readlines()
        lineslist = [x.strip() for x in alllines]
        for line_with_whitespaces in lineslist:
            line = (" ".join(line_with_whitespaces.split())).split()
            if len(line) == 5:
                conceptid = line[0]
                conceptshot = line[2]

                first = conceptshot.split('_')[0]
                second = first.split('shot')[1]

                if second not in double_videos:
                    conceptgroundtruth = line[4]

                    if conceptgroundtruth == '1':
                        if not conceptsList.contains(conceptid):
                            concept = Concept.Concept()
                            concept.name = conceptid
                            concept.shots = []
                            concept.videos = []
                        else:
                            concept = conceptsList.dictionary[conceptid]
                        concept.shots.append(conceptshot)

                        video = conceptshot.split('_')[0].split('shot')[1]
                        if video not in concept.videos:
                            concept.videos.append(video)

                        concept.numberOfShots += 1

                        conceptsList.dictionary[conceptid] = concept
        f.close()
    return conceptsList


def read_xml_tree(filename, conceptsList, videofiles):
    tree = ET.parse(filename)
    root = tree.getroot()

    # get needed video files ids
    needed_videofile_ids = set()
    if conceptsList.sorted is True:
        for concept in conceptsList.concept_list:
            for v in concept.videos:
                needed_videofile_ids.add(v)
    else:
        for k in conceptsList.dictionary:
            concept = conceptsList.dictionary[k]
            for v in concept.videos:
                needed_videofile_ids.add(v)

    # collect all videos from xml
    if videofiles is None:
        videofiles = []

    for child in root:
        # check if it occures in a groundtruth_concept
        if child[0].text in needed_videofile_ids:
            vl = VideoFile.VideoFile()
            vl.id = child[0].text

            fn = child[1].text.split('._-o-_.')

            vl.filename = fn[1]
            vl.source = child[3].text

            vl.filepath = vl.source + '/' + vl.filename

            videofiles.append(vl)

    return videofiles


def export_videofilepaths(filename, videofiles):
    with open(filename, 'w') as f:
        for v in videofiles:
            line = '"' + v.filepath + '"\n'
            f.write(line)
        f.close()


def export_videofileids(filename, videofiles):
    with open(filename, 'w') as f:
        for v in videofiles:
            line = '"' + v.ids + '"\n'
            f.write(line)
        f.close()


def readConceptTxt(filename):
    conceptsList = ConceptsList.ConceptsList()
    conceptsList.dictionary = {}
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=' ')  # create reader
        for line in reader:
            if len(line) == 3:
                concept = Concept.Concept()
                concept.name = line[0]
                concept.shots = []
                concept.videos = []
                linesplit = line[1].split(',')

                for l in linesplit:
                    concept.shots.append(l)
                concept.numberOfShots = len(concept.shots)

                linesplit2 = line[2].split(',')

                for l in linesplit2:
                    concept.videos.append(l)

                conceptsList.dictionary[line[0]] = concept

        f.close()
        conceptsList.createList()

    return conceptsList


def read_shot_xmls():
    path = 'shot_xmls/'
    files = [name for name in os.listdir(path) if os.path.isfile(path + name)]
    shots = {}

    for f in files:
        shots.update(read_shot_xml(path + f))

    return shots


def read_shot_xml(filename):
    shots = {}
    id = ''
    timestamp = ''
    searchForMediaTimepoint = False
    with open(filename, 'r') as f:
        alllines = f.readlines()
        for line in alllines:
            if 'RKF' in line:
                splits = line.split('"')
                without_RKF = splits[1].split('_RKF')
                id = without_RKF[0]
                searchForMediaTimepoint = True

            if '<MediaTimePoint' in line:
                if searchForMediaTimepoint:
                    first = line.split('>')
                    second = first[1].split('<')
                    timestamp = second[0]
                    shot = Shot.Shot(id, timestamp, True)
                    shots[id] = shot
                    searchForMediaTimepoint = False

        f.close()

    return shots


def export_shots(filename, shots):
    with open(filename, 'w') as f:
        for s in shots.values():
            f.write(s.toString() + '\n')
        f.close()


def read_shots_from_file(filename):
    shots = {}
    shot_paths = read_shot_paths('../groundtruth/shot_paths.txt')
    i = 0
    with open(filename, 'r') as f:
        # alllines = f.readlines()
        line = f.readline()
        while line != '':
            splits = line.split(' ')
            if len(splits) > 0:
                shot = Shot.Shot(splits[0], splits[1].split('\n')[0], False)
                # print i
                # shot.shot_path = shot_paths[i]
                shots[shot.name] = shot
                i += 1
            line = f.readline()

        f.close()

    return shots


def read_selected_shots_from_file(filename, conceptsList):
    shots2 = {}

    shots = read_shots_from_file(filename)
    for concept in conceptsList.dictionary.values():
        for shotname in concept.shots:
            shots2[shotname] = shots[shotname]

    return shots2


def export_videofiles(filename, videofiles):
    with open(filename, 'w') as f:
        for v in videofiles:
            f.write(v.toString() + '\n')

        f.close()


def read_videofiles(filename):
    videofiles = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line != '':
            splits = line.split(' ')
            if len(splits) == 4:
                video = VideoFile.VideoFile()
                video.id = splits[0]
                video.filename = splits[1]
                video.source = splits[2]
                video.filepath = splits[3]
                videofiles.append(video)

            line = f.readline()

        f.close()

    return videofiles


def readDoubleVideos(filename):
    double_videos = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line != '':
            splits = line.split('\n')
            double_videos.append(splits[0])

            line = f.readline()

        f.close()

    return double_videos


def export_ffmpeg(filename, ffmpeg_commands):
    with open(filename, 'w') as f:
        for c in ffmpeg_commands:
            f.write(c + '\n')

        f.close()


def export_shot_paths(filename, shot_paths):
    with open(filename, 'w') as f:
        for c in shot_paths:
            f.write(c + '\n')

        f.close()


def read_shot_paths(filename):
    shot_paths = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line != '':
            splits = line.split('\n')
            shot_paths.append(splits[0])
            line = f.readline()

        f.close()

    return shot_paths


def export_metadata_error_files(filename, videofiles):
    with open(filename, 'w') as f:
        for v in videofiles:
            if not os.path.isdir('../videodataset/' + v.id):
                f.write(v.id + '\n')

        f.close()


def read_metadata_error_files(filename):
    data = []
    with open(filename, 'r') as f:
        alllines = f.readlines()
        for line in alllines:
            data.append(line.split('\n')[0])

        f.close()

    return data