
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
    print '\n'
    for line in matrix:
        print line[0]
        for tab in line[1]:
            print '\t%s' % tab
    print '\n'

### create csv of list
def csv_list(list, file_name):
    with open(file_name +'.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter='\t')

        new_line = []
        for a in list:
            new_line.append(a[0])
            new_line.append(a[1])
            writer.writerow(new_line)
            new_line = []

        ### add last line to check the sum of every row is 1.0
        last_line = ['\t']
        tot = 0.0
        for count, a in enumerate(list):
            tot += a[1]
        last_line.append(round(tot,4))
        writer.writerow(last_line)

    print '\tloaded in >> %s.csv <<' %file_name

### create csv of simple matrix
def csv_s_matrix(matrix, file_name):
    with open(file_name +'.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter='\t')

        first_line = ['\t']
        for a in matrix:
            first_line.append(a[0])
        writer.writerow(first_line)

        new_line = []
        for counter, a in enumerate(matrix):
            new_line.append(a[0])
            for x in range(len(matrix)):
                new_line.append(matrix[counter][1][x][1])
            writer.writerow(new_line)
            new_line = []

        ### add last line to check the sum of every row is 1.0
        last_line = ['\t']
        for t in matrix:
            tot = 0.0
            for count, tt in enumerate(t[1]):
                tot += tt[1]
            last_line.append(round(tot,4))
        writer.writerow(last_line)

    print '\tloaded in >> %s.csv <<' %file_name

### create csv of matrix with triple sensors
def csv_t_matrix(matrix, file_name):
    with open(file_name +'.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter='\t')

        first_line = ['\t']
        for a in matrix[0][1]:
            first_line.append(a[0])
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
            for a in t[1]:
                new_line.append(a[1])

            writer.writerow(new_line)
            new_line = []

        ### add last line to check the sum of every row is 1.0
        last_line = ['\t']
        for t in matrix:
            tot = 0.0
            for count, tt in enumerate(t[1]):
                tot += tt[1]
            last_line.append(round(tot,4))
        writer.writerow(last_line)

    print '\tloaded in >> %s.csv <<' %file_name

### check the correctness of start/end times of detections in path_file.txt
### and generate the relative path_file.csv file
def check_and_generate_csv(path_file):
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

                    ### check correctness between end and start in next row
                    # next_smart_line = []
                    # nxt_line = next_line.split('\t')
                    # for colu in nxt_line:
                    #     if colu and not colu.isspace():
                    #         colu = colu.strip()
                    #         next_smart_line.append(colu)
                    # next_start = datetime.strptime(next_smart_line[0], '%Y-%m-%d %H:%M:%S')
                    # if end <= next_start:
                    #     list_detection.append(smart_line)
                    # else:
                    #     error = True
                    #     print '\tstart/end time mismatch in lines %s and %s\n\t%s\n\t%s' %(counter+3, counter+4, smart_line, next_smart_line)

                    ### check correctness between start and start in next row
                    # next_smart_line = []
                    # nxt_line = next_line.split('\t')
                    # for colu in nxt_line:
                    #     if colu and not colu.isspace():
                    #         colu = colu.strip()
                    #         next_smart_line.append(colu)
                    # next_start = datetime.strptime(next_smart_line[0], '%Y-%m-%d %H:%M:%S')
                    # if start <= next_start:
                    #     list_detection.append(smart_line)
                    # else:
                    #     error = True
                    #     print '\tstart/end time mismatch in lines %s and %s\n\t%s\n\t%s' %(counter+3, counter+4, smart_line, next_smart_line)

                    list_detection.append(smart_line)
                else:
                    error = True
                    print '\tstart/end time mismatch at line %s\n\t %s' %(counter+3, smart_line)
    if not error:
        with open(path_file +'.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
            for line in list_detection:
                writer.writerow(line)
        print '\tcorrectly loaded in >> %s.csv <<\n' %path_file
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
def p_adls(path_file, house_name):
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
    print '%s: p(adls) calculated' %house_name
    csv_list(list_adls, house_name+'_p(adls)')
    return list_adls

### probability of going from adl at time (t-1) to adl at time (t)
### p(y(t)|y(t-1)) = p(adl(t)|adl(t-1))
def p_adls_from_adls(path_file, list_adls, house_name):
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
        normalize_matrix(matrix)

    print '%s : p(adls|adls) calculated' %house_name
    csv_s_matrix(matrix, house_name+'_p(adls|adls)')
    del list_adls, matrix

def t_sens(path_file):
    ### build list of all possible triple Location-Type-Place
    list_t_sens = []
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        for row in reader:
            triple = [copy.deepcopy(row[2]), copy.deepcopy(row[3]), copy.deepcopy(row[4])]
            exist_triple = False
            for t in list_t_sens:
                if (t[0]==triple[0]) and (t[1]==triple[1]) and (t[2]==triple[2]):
                    exist_triple = True
            if not exist_triple:
                list_t_sens.append(triple)
    return list_t_sens

### probability of adl generate observation of sensor
### an observation of sensor is represented by a triple of Location-Type-Place
### p(x|y) = p(sens|adl)
def p_sens_from_adls(path_file_adls, list_adls, path_file_sens, list_t_sens, house_name):

    ### initialize list_adls probability to 0
    for a in list_adls:
        a[1] = 0

    ### initialize matrix:
    ###   follow each sens formed by a triple (t) with all possible adls (a)
    ###   and for each of them the probability of <t generated by a>
    matrix = []
    for t in list_t_sens:
        t = [copy.deepcopy(t), copy.deepcopy(list_adls)]
        matrix.append(t)

    with open(path_file_adls+'.csv', 'rb') as csv_adls:
        reader_adls = csv.reader(csv_adls, dialect='excel', delimiter='\t')

        for ca, a in enumerate(reader_adls):
            adl = a[2]
            start_a = datetime.strptime(a[0], '%Y-%m-%d %H:%M:%S')
            end_a = datetime.strptime(a[1], '%Y-%m-%d %H:%M:%S')
            # print '%s: %s - %s' %(adl, start_a, end_a)

            with open(path_file_sens+'.csv', 'rb') as csv_sens:
                reader_sens = csv.reader(csv_sens, dialect='excel', delimiter='\t')

                for ct, t in enumerate(reader_sens):
                    t_sens = [t[2], t[3], t[4]]
                    start_t = datetime.strptime(t[0], '%Y-%m-%d %H:%M:%S')
                    end_t = datetime.strptime(t[1], '%Y-%m-%d %H:%M:%S')

                    if (start_t >= start_a) and (end_t <= end_a):
                        # print '\t%s\n\t %s - %s' %(t_sens, start_t, end_t)
                        increment_sens_from_adls(matrix, t_sens, adl)

    normalize_matrix(matrix)
    # print_matrix(matrix)
    print '%s : p(sens|adls) calculated' %house_name
    csv_t_matrix(matrix, house_name+'_p(sens|adls)')
    del list_t_sens, matrix


def increment_sens_from_adls(matrix, t_sens, adl):
    for t in matrix:
        if t[0] == t_sens:
            for a in t[1]:
                if a[0] == adl:
                    a[1] += 1

def normalize_matrix(matrix):
    for t in matrix:
        tot_occ = 0
        for a in t[1]:
            tot_occ += a[1]
        if tot_occ != 0:
            part_occ = 0
            for count, a in enumerate(t[1]):
                if count+1 == len(t[1]):
                    a[1] = round(1.00 - part_occ, 4)
                else:
                    a[1] = round(numpy.divide(float(a[1]), float(tot_occ)), 4)
                    part_occ += a[1]

def project():

    ### dataset is the list of each house analyzed, in each house:
    ###     house[0] = name house
    ###     house[1] = Description
    ###     house[2] = ADLs
    ###     house[3] = Sensors

    # ordonezA = ['Dataset/OrdonezA','Dataset/OrdonezA_Description','Dataset/OrdonezA_ADLs','Dataset/OrdonezA_Sensors']
    # ordonezA = ['Dataset/OrdonezB','Dataset/OrdonezB_Description','Dataset/OrdonezB_ADLs','Dataset/OrdonezB_Sensors']
    # dataset = [ordonezA, ordonezB]

    test1 = ['Deposet/test1','Deposet/test1_Description','Deposet/test1_ADLs','Deposet/test1_Sensors']
    test2 = ['Deposet/test2','Deposet/test2_Description','Deposet/test2_ADLs','Deposet/test2_Sensors']
    dataset = [test1, test2]

    for house in dataset:
        print '\n'
        for path_file in house:
            ### check the correctness of file
            if ('ADLs' in path_file) or ('Sensors' in path_file):
                check_and_generate_csv(path_file)
        house_name = house[0]
        path_adls = house[2]
        ### obtain the list adls
        list_adls = p_adls(path_adls, house_name)
        ### calculate p(adls|adls)
        p_adls_from_adls(path_adls, list_adls, house_name)

        path_sens = house[3]
        ### obtain the list sens
        list_sens = t_sens(path_sens)
        ### calculate p(sens|adls)
        p_sens_from_adls(path_adls, list_adls, path_sens, list_sens, house_name)
        print '\n------------------------------------------------------------'

if __name__ == '__main__':
    project()
