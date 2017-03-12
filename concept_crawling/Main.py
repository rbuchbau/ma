
import FileIO
import ConceptsList
import timeit

def main():

    time1 = timeit.default_timer()

    # read concepts from labels files
    conceptsList = ConceptsList.ConceptsList()
    conceptsList = FileIO.read_labels('groundtruths/labels.txt', conceptsList)
    conceptsList = FileIO.read_labels('groundtruths/labels2.txt', conceptsList)

    time2 = timeit.default_timer()
    print "Time: " + str(time2 - time1) + " seconds"

    # write to csv file
    # fileout = open('concepts.txt', 'w')
    # fileout.write(conceptsList.toString())
    # fileout.close()

    # sort concepts by occurences
    conceptsList.sortList()
    for c in conceptsList.concept_list:
        print c.name + ' ' + str(c.numberOfShots) + ' ' + str(len(c.videos))

    print len(conceptsList.dictionary)

    ids = ['1267', '1005', '1015', '1261', '1031', '1010', '1006']
    conceptsList = conceptsList.copyNConcepts(ids)

    # read xml
    videofiles = FileIO.read_xml_tree('videolists/video_src1.xml', conceptsList, None)
    videofiles = FileIO.read_xml_tree('videolists/video_src2.xml', conceptsList, videofiles)
    videofiles = FileIO.read_xml_tree('videolists/video_src3.xml', conceptsList, videofiles)

    print len(videofiles)

    FileIO.export_videofilepaths('download_videofiles_selected.txt', videofiles)




    print " "




if __name__ == '__main__':
    main()