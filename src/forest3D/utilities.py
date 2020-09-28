'''
    Author: Dr. Lloyd Windrim
    Required packages: numpy

    Simple utlility functions e.g. reading and writing csv's

'''

import csv
import numpy as np
import matplotlib.pyplot as plt

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


def write_csv(filename,data,header=None,delimiter=','):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter=delimiter,quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([header])
        for i in range(np.shape(data)[0]):
            spamwriter.writerow(data[i,:])



class plot_vars:

    def __init__(self, var_list, save_addr):

        self.var_list = var_list
        self.save_addr = save_addr
        self.epochs = []

        self.vars = {}
        for i,var_name in enumerate(self.var_list):
            self.vars[var_name] = []

    def update_plot(self, epoch, new_vars):
        self.epochs.append(epoch)
        plt.figure()
        for i, var_name in enumerate(self.var_list):
            self.vars[var_name].append(new_vars[i])
            plt.plot(self.epochs, self.vars[var_name], label=self.var_list[i])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.save_addr)
        plt.close()

