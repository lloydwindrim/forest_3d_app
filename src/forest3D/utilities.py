import csv
import numpy as np

def read_csv(filename,header_rows=[-1]):

    data = []
    data_dict = {}
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i,row in enumerate(spamreader):
            if i in header_rows:
                data_dict['header'] = row
            else:
                data.append((np.array(row)).astype(np.float))

        data = np.array(data)

        data_dict['num_entries'] = np.shape(data)[0]
        data_dict['data'] = data

        return data_dict


def write_csv(filename,data,header=None):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([header])
        for i in range(np.shape(data)[0]):
            spamwriter.writerow(data[i,:])





