import csv
import xml.etree.ElementTree as ET

import Concept
import ConceptsList
import VideoFile


# read ground-truth
def read_labels(filename, conceptsList):
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
