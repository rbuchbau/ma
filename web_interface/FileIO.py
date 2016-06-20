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

#convert groundtruth to higher resolution (eg. 2secs -> 0.5secs: factor = 4)
def convert_groundtruth(data, filename, factor):
    with open('groundtruths/' + filename, 'w') as f:
        f.write("# sport or watersport\n# startimage,endimage\n")
        for a,b in data:
            a = int(math.ceil(float(a)/factor))
            b = int(math.floor(float(b)/factor))
            f.write(str(a) + "," + str(b) + "\n")
        f.close()

