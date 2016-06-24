import csv

#read ground-truth
def read_groundtruth(filename):
    data = []
    with open(filename, 'r') as f:    #open file for reading
        reader = csv.reader(f, delimiter=' ')           #create reader
        for line in reader:
            if len(line) == 2:
                data.append( (line[0], int(line[1])) )
        f.close()
    return data