import csv

import numpy as np

#read ground-truth
def read_synsets():
    data = []
    with open('synsets.csv', 'r') as f:                  #open file for reading
        reader = csv.reader(f)       #create reader
        for line in reader:
            data.append(line[0])
    return data


# #write histogram csvs with numpy
# def save_histograms_to_file(filename, mdh_all):
#     np.savetxt('motionDirectionHistograms/' + filename, mdh_all, delimiter=',', fmt='%4.4f')


#read csvs with numpy
def read_csv(filename, delimiter=" "):
    data = np.genfromtxt(filename, dtype='|S', delimiter=delimiter)
    return data

