from fileinput import filename
import os
import re
import csv

path_to_images = 'C:/Users/matan/Documents/data/coil-20-proc/coil-20-proc'
path_to_csv = 'C:/Users/matan/Documents/data/coil-20-proc/labels.csv'
filenames = os.listdir(path_to_images)

with open(path_to_csv, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(('image', 'label'))
    for name in filenames:
        assert name[0:3] == 'obj'
        assert name[-4:] == '.png'
        numbers_in_filename = re.findall(r'\d+', name)
        label = numbers_in_filename[0]

        writer.writerow((name, label))
