import csv
import math

#read ground-truth
def read_groundtruth(filename):
    data = []
    with open('groundtruths/' + filename, 'r') as f:    #open file for reading
        reader = csv.reader(f, delimiter=',')           #create reader
        reader.next()
        reader.next()
        for line in reader:
            if len(line) == 2:
                data.append( (int(line[0]), int(line[1])) )
        f.close()
    return data


def read_csv(filename):
    data = []
    with open(filename, 'r') as f:    #open file for reading
        reader = csv.reader(f, delimiter=' ')           #create reader
        for line in reader:
            if len(line) == 2:
                data.append( (line[0], int(line[1])) )
        f.close()
    return data


def read_synset_words(filename):
    data = []
    with open(filename, 'r') as f:    #open file for reading
        reader = csv.reader(f, delimiter=' ')           #create reader
        for i,line in enumerate(reader):
            str = '' + line[1]
            for j,string in enumerate(line):
                if j > 1:
                    str += ' ' + string
            data.append(str)
        f.close()
    return data


#read ground-truth
def read_groundtruth(video_name):
    data = []
    with open('groundtruths/' + video_name, 'r') as f:    #open file for reading
        reader = csv.reader(f, delimiter=',')           #create reader
        reader.next()
        reader.next()
        for line in reader:
            if len(line) == 2:
                data.append( (int(line[0]), int(line[1])) )
        f.close()
    return data

#convert groundtruth to higher resolution (eg. 2secs -> 0.5secs: factor = 4)
def convert_groundtruth(data, filename, factor):
    with open('groundtruths/' + filename, 'w') as f:
        f.write("# sport or watersport\n# startimage,endimage\n")
        for a,b in data:
            a = int(math.ceil(float(a)/factor))
            b = int(math.floor(float(b)/factor))
            f.write(str(a) + "," + str(b) + "\n")
        f.close()


def convert_groundtruths(videos):
    for v in videos:
        video_name = v + '/' + 'groundtruth_' + v
        data = read_groundtruth(video_name + '_05secs.csv')
        convert_groundtruth(data, video_name + '_1secs.csv', 2)
        convert_groundtruth(data, video_name + '_2secs.csv', 4)
        convert_groundtruth(data, video_name + '_3secs.csv', 6)
        convert_groundtruth(data, video_name + '_4secs.csv', 8)
        convert_groundtruth(data, video_name + '_5secs.csv', 10)


def write_times(filename, time, svm_size, kernel):
    with open(filename, 'a') as f:
        f.write(kernel + "," + str(svm_size) + "," + str(time) + "\n")
        f.close()
