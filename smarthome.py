# from __future__ import division
# import os as os
# import scipy as scipy
import csv
import copy
from datetime import datetime
import numpy as numpy
# import ghmm


### print each line of input collection
def print_csv(path_file):
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        for row in reader:
            print ' | '.join(row)

### print matrix structure
def print_matrix(matrix):
    print '================================================================='
    for line in matrix:
        print line[0]
        for tab in line[1]:
            print '\t%s' % tab
    print '================================================================='

### check the correctness of start/end times of detections in path_file.txt
### and generate the relative path_file.csv file
def check_and_generate_csv(path_file):
    print '-----------------------------------------------------------------'
    print 'check file >> %s.txt <<' %path_file
    file_input = open(path_file+'.txt').readlines()
    list_detection = []
    error = False
    iter_detection = iter(file_input)
    try:
        ### two times next() to jump first and second line in file (header of table)
        next(iter_detection)
        next(iter_detection)
        next_line = next(iter_detection)
        init_error = False
        ### to count lines in file
        for i, l in enumerate(file_input):
            pass
        tot_lines = i-1
    except StopIteration:
        init_error = True
    if not init_error:
        for counter in range(tot_lines):
            smart_line = []
            this_line = next_line.split('\t')
            try:
                next_line = next(iter_detection)
                next_empty = False
            except StopIteration:
                next_empty = True
            if not next_empty:
                for col in this_line:
                    if col and not col.isspace():
                        col = col.strip()
                        smart_line.append(col)
                start = datetime.strptime(smart_line[0], '%Y-%m-%d %H:%M:%S')
                end = datetime.strptime(smart_line[1], '%Y-%m-%d %H:%M:%S')
                if start <= end:
                    next_smart_line = []
                    nxt_line = next_line.split('\t')
                    for colu in nxt_line:
                        if colu and not colu.isspace():
                            colu = colu.strip()
                            next_smart_line.append(colu)
                    next_start = datetime.strptime(next_smart_line[0], '%Y-%m-%d %H:%M:%S')
                    if end <= next_start:
                        list_detection.append(smart_line)
                    else:
                        error = True
                        print '\tstart/end time mismatch in lines %s and %s\n\t%s\n\t%s' %(counter+3, counter+4, smart_line, next_smart_line)
                else:
                    error = True
                    print '\tstart/end time mismatch at line %s\n\t %s' %(counter+3, smart_line)
    if not error:
        with open(path_file +'.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
            for line in list_detection:
                writer.writerow(line)
        print '\tcorrectly loaded in >> %s.csv <<' %path_file
        print '-----------------------------------------------------------------\n'
        del list_detection, iter_detection
    else:
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        print '[!] there are errors in file:'
        print '[!] \t>> %s.txt <<' % path_file
        print '[!] please fix them before continuing.'
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
        del list_detection, iter_detection
        quit()

### probability of each adl
### p(y) = p(adl)
def p_adls(path_file):
    tot_rows = len(open(path_file+'.csv').readlines())
    ### build list of all possible adls
    list_adls = []
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        ### count how many times adl compare
        for row in reader:
            adl = [copy.deepcopy(row[2]), 1]
            exist_adl = False
            for a in list_adls:
                if (a[0] == adl[0]):
                    a[1] += 1
                    exist_adl = True
            if not exist_adl:
                list_adls.append(adl)
        ### normalize
        partial_tot = 0
        for count, a in enumerate(list_adls):
            if count+1 == len(list_adls):
                a[1] = round(1.00 - partial_tot, 4)
            else:
                a[1] = round(numpy.divide(float(a[1]), float(tot_rows)), 4)
                partial_tot += a[1]
    return list_adls

### probability of adl at time (t) after adl at time (t-1)
### p(y(t)|y(t-1)) = p(adl(t)|adl(t-1))
def p_adls_cond(path_file, list_adls):
    tot_rows = len(open(path_file+'.csv').readlines())

    ### initialize list_adls probability to 0
    for a in list_adls:
        a[1] = 0

    ### initialize matrix:
    ###   follow each adl (a) with all possible adls (aa)
    ###   and for each of them the probability of <a followed by aa>
    matrix = []
    for a in list_adls:
        a = [copy.deepcopy(a[0]), copy.deepcopy(list_adls)]
        matrix.append(a)

    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        ### count how many times adl followed by other adl
        iter_file = iter(reader)
        nxt = next(iter_file)
        for counter in range(tot_rows):
            this_a = nxt[2]
            try:
                nxt = next(iter_file)
                next_empty = False
            except StopIteration:
                next_empty = True
            if not next_empty:
                next_a = nxt[2]
                for a in matrix:
                    if a[0] == this_a:
                        for aa in a[1]:
                            if aa[0] == next_a:
                                aa[1] += 1
        ### normalize
        for a in matrix:
            tot_occ = 0
            for aa in a[1]:
                tot_occ += aa[1]
            if tot_occ != 0:
                part_occ = 0
                for count, aa in enumerate(a[1]):
                    if count+1 == len(a[1]):
                        aa[1] = 1.00 - part_occ
                    else:
                        aa[1] = numpy.divide(float(aa[1]), float(tot_occ))
                        part_occ += aa[1]

    # print_matrix(matrix)
    csv_matrix(matrix, path_file+'_p(adls|adls)')
    del list_adls, matrix

### create matrix with probability of events in Sensors files
def matrix_from_sensors(path_file):

    tot_rows = len(open(path_file+'.csv').readlines())

    ### build list of all possible triple Location-Type-Place
    list_triple = []
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        for row in reader:
            triple = [copy.deepcopy(row[2]), copy.deepcopy(row[3]), copy.deepcopy(row[4])]
            exist_triple = False
            for t in list_triple:
                if (t[0]==triple[0]) and (t[1]==triple[1]) and (t[2]==triple[2]):
                    exist_triple = True
            if not exist_triple:
                list_triple.append(triple)

    ### build support list with weight (initialized to 0) of each triple
    list_weight_triple = []
    for t in list_triple:
        t = [copy.depcopy(t), 0]
        list_weight_triple.append(t)

    ### initialize matrix with triple weight event:
    ###   follow each triple (t) with all possible triple (tt)
    ###   and for each of them the probability of <t followed by tt>
    matrix = []
    for t in list_triple:
        t = [copy.deepcopy(t), copy.deepcopy(list_weight_triple)]
        matrix.append(t)

    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        iter_file = iter(reader)
        nxt = next(iter_file)
        for counter in range(tot_rows):
            this_triple = [nxt[2], nxt[3], nxt[4]]
            try:
                nxt = next(iter_file)
                next_empty = False
            except StopIteration:
                next_empty = True
            if not next_empty:
                next_triple = [nxt[2], nxt[3], nxt[4]]
                for t in matrix:
                    if t[0] == this_triple:
                        for tt in t[1]:
                            if tt[0] == next_triple:
                                tt[1] += 1

        for t in matrix:
            tot_occ = 0
            for tt in t[1]:
                tot_occ += tt[1]
            if tot_occ != 0:
                part_occ = 0
                for count, tt in enumerate(t[1]):
                    if count+1 == len(t[1]):
                        tt[1] = 1.00 - part_occ
                    else:
                        tt[1] = numpy.divide(float(tt[1]), float(tot_occ))
                        part_occ += tt[1]

        # print_matrix(matrix)
        csv_matrix(matrix, path_file+'_matrix')
        del list_triple, list_weight_triple, matrix

### create csv of my matrix
def csv_matrix(matrix, file_name):
    with open(file_name +'.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter='\t')

        first_line = ['\t']
        for t in matrix:
            triple = ''
            for count, string in enumerate(t[0]):
                if count+1 == len(t[0]):
                    triple += string
                else:
                    triple += string + ' '
            first_line.append(triple)
        writer.writerow(first_line)

        new_line = []
        for counter, t in enumerate(matrix):
            triple = ''
            for count, string in enumerate(t[0]):
                if count+1 == len(t[0]):
                    triple += string
                else:
                    triple += string + ' '
            new_line.append(triple)
            for x in range(len(matrix)):
                new_line.append(matrix[x][1][counter][1])

            writer.writerow(new_line)
            new_line = []

        ### add last line to check the sum of every row is 1.0
        # last_line = ['\t']
        # for t in matrix:
        #     tot = 0.0
        #     for count, tt in enumerate(t[1]):
        #         tot += tt[1]
        #     last_line.append(tot)
        # writer.writerow(last_line)

    ### print matrix created in csv file
    # with open(file_name +'.csv', 'rb') as csvfile:
    #     reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
    #     for line in reader:
    #         print line

def project():
    # dataset = ['Dataset/OrdonezA_ADLs','Dataset/OrdonezA_Sensors', 'Dataset/OrdonezB_ADLs','Dataset/OrdonezB_Sensors']
    # dataset = ['Dataset/Sensors-test', 'Dataset/Sensors-test2', 'Dataset/Sensors-test3',]
    dataset = ['Dataset/ADLs-test1',]
    for path_file in dataset:
        check_and_generate_csv(path_file)
        # print_csv(path_file)
        if 'Sensors' in path_file:
            matrix_from_sensors(path_file)
        if 'ADLs' in path_file:
            list_adls = p_adls(path_file)
            p_adls_cond(path_file, list_adls)

    print 'ciao Depo :)'

if __name__ == '__main__':
    project()
