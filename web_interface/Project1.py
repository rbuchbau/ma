import HTML
import XML
import FileIO


#main program
def main():


    #html stuff:
    #-------------------------------------------------------------------
    #HTML.html_stuff('./html_input/index_part1.txt', './html_input/index_part2.txt', './html_output/index.html')
    #-------------------------------------------------------------------


    #convert groundtruth from 0.5secs intervall (manual input) to usable intervals (1,2,3,4,5secs)
    #data = FileIO.read_groundtruth('groundtruth_1001_05secs.csv')
    #FileIO.convert_groundtruth(data, 'groundtruth_1001_2secs.csv', 4)


    #xml stuff:
    #-------------------------------------------------------------------
    XML.create_XML('caffe_output_2secs_p_c_3.txt', 'groundtruth_1001_2secs.csv', '1001_77_2secs_p_c.csv')
    #-------------------------------------------------------------------


if __name__ == '__main__':
    main()
