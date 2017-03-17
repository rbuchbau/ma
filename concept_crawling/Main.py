
import FileIO
import Folders
import ConceptsList
import timeit
import os
import Shot

def main():

    # read double videos
    double_videos = FileIO.readDoubleVideos('double_elements.txt')

    #create all
    # create list of concepts
    # ids = ['1267', '1005', '1015', '1261', '1031', '1010', '1006']
    # conceptsList = createConceptsList(ids, double_videos)
    # # create list of videos
    # createVideofiles(conceptsList)


    # or read them from csv
    conceptsList = FileIO.readConceptTxt('concepts.txt')
    videofiles = FileIO.read_videofiles('needed_videos.txt')


    # create folders and move videofiles, also check for double videos and export them
    path = 'videodataset/'
    # createFolders(videofiles, path)



    # shots = FileIO.read_shot_xmls()
    # FileIO.export_shots('shots.csv', shots)
    needed_shots = FileIO.read_selected_shots_from_file('shots.csv', conceptsList)

    ffmpeg_commands = createFFMPEGCommands(needed_shots, path)
    FileIO.export_ffmpeg('ffmpeg_commands.txt', ffmpeg_commands)


    print " "


def createConceptsList(ids, double_videos):
    time1 = timeit.default_timer()

    # read concepts from labels files
    conceptsList = ConceptsList.ConceptsList()
    conceptsList = FileIO.read_labels('groundtruths/labels.txt', conceptsList, double_videos)
    conceptsList = FileIO.read_labels('groundtruths/labels2.txt', conceptsList, double_videos)

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
    commands = []
    videodirs = [name for name in os.listdir(path) if os.path.isdir(path + name)]
    for v in videodirs:
        for shot in needed_shots.values():
            if v == shot.video:
                commands.append('ffmpeg -ss ' + shot.timestamp + ' -i ' + \
                      shot.video + '/' + shot.video + '.mp4 -t 0.04 ' + \
                      shot.video + '/%03d.jpg;')

    return commands


if __name__ == '__main__':
    main()