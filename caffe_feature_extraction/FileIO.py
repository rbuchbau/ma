import csv
import math
import ConceptMeasurements


# read ground-truth
def read_groundtruth(filename):
    data = []
    path = 'groundtruths/' + filename
    print "Read groundtruth: " + path
    with open(path, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=',')  # create reader
        reader.next()
        reader.next()
        for line in reader:
            if len(line) == 2:
                data.append((int(line[0]), int(line[1])))
        f.close()
    return data


# read ground-truth
def read_groundtruth_with_header(filename):
    data = []
    with open('groundtruths/' + filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=',')  # create reader
        data.append(reader.next())
        reader.next()
        for line in reader:
            if len(line) == 2:
                data.append((int(line[0]), int(line[1])))
        f.close()
    return data


def read_csv(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=' ')  # create reader
        for line in reader:
            if len(line) == 2:
                data.append((line[0], int(line[1])))
        f.close()
    return data


def read_synset_words(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=' ')  # create reader
        for i, line in enumerate(reader):
            str = '' + line[1]
            for j, string in enumerate(line):
                if j > 1:
                    str += ' ' + string
            data.append(str)
        f.close()
    return data


# convert groundtruth to higher resolution (eg. 2secs -> 0.5secs: factor = 4)
def convert_groundtruth(data, filename, factor):
    with open('groundtruths/' + filename, 'w') as f:
        for i, x in enumerate(data):
            if i == 0:
                f.write(x[0] + " \n# startimage, endimage\n")
            else:
                (a, b) = x
                a = int(math.ceil(float(a) / factor))
                b = int(math.floor(float(b) / factor))
                f.write(str(a) + "," + str(b) + "\n")
        f.close()


def convert_groundtruths(videos):
    for v in videos:
        video_name = v + '/' + 'groundtruth_' + v
        data = read_groundtruth_with_header(video_name + '_05secs.csv')
        convert_groundtruth(data, video_name + '_1secs.csv', 2)
        convert_groundtruth(data, video_name + '_2secs.csv', 4)
        convert_groundtruth(data, video_name + '_3secs.csv', 6)
        convert_groundtruth(data, video_name + '_4secs.csv', 8)
        convert_groundtruth(data, video_name + '_5secs.csv', 10)


def write_times(filename, time, model, feature):
    with open(filename, 'a') as f:
        f.write(model + "_" + str(feature) + "," + str(time) + "\n")
        f.close()


def write_accuracies(all_concepts, model):
    path = 'acc_results/'
    with open(path + model + '.txt', 'w') as f:
        for concept_measurement in all_concepts.values():
            f.write(concept_measurement.toString() + '\n')

        f.close()


def read_accuracies_and_average_them(filename):
    lengths = [0, 0, 0]

    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=' ')
        for line in reader:
            if len(line) == 7:
                # name = line[0]
                # precision = line[1]
                # recall = line[2]
                # f_measure = line[3]

                lengths[0] += int(line[4])
                lengths[1] += int(line[5])
                lengths[2] += int(line[6])

        f.close()

    return lengths


def write_average_accuracies(filename, models):

    for model in models:
        values = [0, 0, 0]
        (a, b, c) = read_accuracies_and_average_them('acc_results/' + model + '.txt')
        values[0] = a
        values[1] = b
        values[2] = c

        prec, rec, f_m = 0.0, 0.0, 0.0

        if values[1] > 0:
            prec = float(values[0]) / values[1]

        if values[1] > 0:
            rec = float(values[0]) / values[2]

        if (prec + rec) > 0:
            f_m = 2 * float(prec * rec) / (prec + rec)

        with open(filename, 'a') as f:
            f.write(str(prec) + ' ' + str(rec) + ' ' + str(f_m) + ' ' + model + '\n')

            f.close()
