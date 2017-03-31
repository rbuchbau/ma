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
        f.write(model + "_" + str(feature) + " " + str(time) + "\n")
        f.close()


def write_accuracies(all_concepts, filename):
    path = 'acc_results/'
    with open(path + filename + '.txt', 'w') as f:
        for concept_measurement in all_concepts.values():
            f.write(concept_measurement.toString() + '\n')

        f.close()


def read_accuracies_and_average_them(filename):
    lengths = [0, 0, 0]

    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=' ')
        for line in reader:
            if len(line) == 7:
                lengths[0] += int(line[4])
                lengths[1] += int(line[5])
                lengths[2] += int(line[6])

        f.close()

    return lengths


def write_average_accuracies(filename, model):
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
        f.write(model + ' ' + "{0:.4f}".format(prec) + ' ' + "{0:.4f}".format(rec) + ' ' + "{0:.4f}".format(f_m) + '\n')

        f.close()


def format_accuracies(filename_from, filename_to, filename_concepts, modfeat):
    data = []
    with open(filename_from, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=' ')  # create reader
        for line in reader:
            if len(line) == 7:
                data.append( (line[0], line[1], line[2], line[3], line[4], line[5], line[6]) )
        f.close()

    #sort
    data_sorted = sorted(data, key=lambda tup: tup[0])

    with open(filename_to, 'a') as f_w:
        f_w.write('Concept Precis. Recall F-Meas. #TP #Pos #Rel\n')
        for d in data_sorted:
            #output per model/feature
            (concept, prec, rec, f_m, tp, po, rel) = d
            f_w.write(concept + ' ' + "{0:.4f}".format(float(prec)) + ' ' + "{0:.4f}".format(float(rec)) + ' ' +
                      "{0:.4f}".format(float(f_m)) + ' ' + tp + ' ' + po + ' ' + rel + '\n')

            #output per concept
            with open(filename_concepts + concept + '.txt', 'a') as f_c:
                f_c.write(modfeat + ' ' + "{0:.4f}".format(float(prec)) + ' ' + "{0:.4f}".format(float(rec)) + ' ' +
                          "{0:.4f}".format(float(f_m)) + ' ' + tp + ' ' + po + ' ' + rel + '\n')

                f_c.close()


        f_w.close()


def modify_train(filename_in, filename_out, concept):
    with open(filename_out, 'w') as f_o:
        with open(filename_in, 'r') as f_i:  # open file for reading
            reader = csv.reader(f_i, delimiter=' ')  # create reader
            for line in reader:
                if len(line) == 2:
                    conc = line[1]
                    if conc == str(concept):
                        conc = '0'
                    else:
                        conc = '1'

                    f_o.write(line[0] + ' ' + conc + '\n')
            f_i.close()
        f_o.close()


def modify_synset_words(filename_in, filename_out, id, concept):
    data = []
    with open(filename_in, 'r') as f_i:  # open file for reading
        lines = f_i.readlines()
        for line in lines:
            splits = line.split('\n')
            split = splits[0]
            cid = split[0:9]
            conc = split[10:]

            if conc != concept:
                conc = 'other'

            data.append( (cid, conc) )

        f_i.close()

    # rearranging so the searched concept has always id 0 and the rest id 1
    data2 = []
    data2.append(data[int(id)])
    del data[int(id)]
    data2.extend(data)

    with open(filename_out, 'w') as f_o:
        for d in data2:
            a, b = d
            f_o.write(a + ' ' + b + '\n')

        f_o.close()
