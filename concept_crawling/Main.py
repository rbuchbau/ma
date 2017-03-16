
import FileIO
import Folders
import ConceptsList
import timeit
import os
import Shot

def main():

    # create list of concepts
    # ids = ['1267', '1005', '1015', '1261', '1031', '1010', '1006']
    # conceptsList = createConceptsList()

    # or read them from csv
    conceptsList = FileIO.readConceptTxt('concepts.txt')

    path = 'videodataset/'

    # create list of videos
    # createVideofiles(conceptsList)

    # or read from csv
    videofiles = FileIO.read_videofiles('needed_videos.txt')
    createFolders(videofiles, path)


    # shots = FileIO.read_shot_xmls()
    # FileIO.export_shots('shots.csv', shots)
    needed_shots = FileIO.read_selected_shots_from_file('shots.csv', conceptsList)

    path = 'videodataset/'
    createFFMPEGCommands(needed_shots, path)



    print " "


def createConceptsList(ids):
    time1 = timeit.default_timer()

    # read concepts from labels files
    conceptsList = ConceptsList.ConceptsList()
    conceptsList = FileIO.read_labels('groundtruths/labels.txt', conceptsList)
    conceptsList = FileIO.read_labels('groundtruths/labels2.txt', conceptsList)

    time2 = timeit.default_timer()
    print "Time: " + str(time2 - time1) + " seconds"


    # sort concepts by occurences
    conceptsList.sortList()
    for c in conceptsList.concept_list:
        print c.name + ' ' + str(c.numberOfShots) + ' ' + str(len(c.videos))

    print len(conceptsList.dictionary)

    conceptsList = conceptsList.copyNConcepts(ids)

    # write to csv file
    fileout = open('concepts.txt', 'w')
    fileout.write(conceptsList.toString())
    fileout.close()

    return conceptsList


def createVideofiles(conceptsList):
    # read xml
    videofiles = FileIO.read_xml_tree('videolists/video_src1.xml', conceptsList, None)
    videofiles = FileIO.read_xml_tree('videolists/video_src2.xml', conceptsList, videofiles)
    videofiles = FileIO.read_xml_tree('videolists/video_src3.xml', conceptsList, videofiles)

    # FileIO.export_videofilepaths('download_videofiles_selected.txt', videofiles)

    # write to csv file
    FileIO.export_videofiles('needed_videos.txt', videofiles)


def createFolders(videofiles, path):
    vorh_videos = [name for name in os.listdir(path)
                   if (os.path.isfile(path + name) and name != 'files.txt')
                   ]
    Folders.createFolders(videofiles, vorh_videos)
    Folders.moveFiles(videofiles, vorh_videos)


def createFFMPEGCommands(needed_shots, path):
    videodirs = [name for name in os.listdir(path) if os.path.isdir(path + name)]
    for v in videodirs:
        for shot in needed_shots.values():
            if v == shot.video:
                print 'ffmpeg -ss ' + shot.timestamp + ' -i ' + \
                      shot.video + '/' + shot.video + '.mp4 -t 0.04 ' + \
                      shot.video + '/%03d.jpg;'



if __name__ == '__main__':
    main()