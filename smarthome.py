
import csv
import copy
from datetime import datetime
import numpy as np


### print matrix structure
def print_matrix(matrix):
    for line in matrix:
        print line

### create csv from lists
def csv_list(list_names, list_values, file_name):
    if (len(list_names) != len(list_values)):
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        print '[!] error in method'
        print '[!] csv_list(list_names, list_values, file_name)'
        print '[!] \t-> len(list_names) != len(list_values)'
        print '[!] please fix before continuing.'
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
        quit()
    else:
        with open(file_name +'.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
            first_line = []
            for x in list_names:
                first_line.append(x)
            writer.writerow(first_line)
            new_line = []
            for y in list_values:
                new_line.append(y)
            writer.writerow(new_line)
        print '\tloaded in >> %s.csv <<' %file_name

### create csv from square matrix
def csv_square_matrix(list_names, matrix, file_name):
    if (len(list_names) != len(matrix)):
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        print '[!] error in method'
        print '[!] csv_square_matrix(list_names, matrix, file_name)'
        print '[!] \t-> len(list_names) != len(matrix)'
        print '[!] please fix before continuing.'
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
        quit()
    else:
        with open(file_name +'.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
            first_line = ['\t']
            for x in list_names:
                first_line.append(x)
            writer.writerow(first_line)
            new_line = []
            for c, y in enumerate(list_names):
                new_line.append(y)
                for z in range(len(matrix)):
                    new_line.append(matrix[c][z])
                writer.writerow(new_line)
                new_line = []
        print '\tloaded in >> %s.csv <<' %file_name

### create csv of matrix with triple sensors
# def csv_t_matrix(matrix, file_name):
#     with open(file_name +'.csv', 'wb') as csvfile:
#         writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
#
#         first_line = ['\t']
#         for a in matrix[0][1]:
#             first_line.append(a[0])
#         writer.writerow(first_line)
#
#         new_line = []
#         for counter, t in enumerate(matrix):
#             triple = ''
#             for count, string in enumerate(t[0]):
#                 if count+1 == len(t[0]):
#                     triple += string
#                 else:
#                     triple += string + ' '
#             new_line.append(triple)
#             for a in t[1]:
#                 new_line.append(a[1])
#
#             writer.writerow(new_line)
#             new_line = []
#
#         ### add last line to check the sum of every row is 1.0
#         last_line = ['\t']
#         for t in matrix:
#             tot = 0.0
#             for count, tt in enumerate(t[1]):
#                 tot += tt[1]
#             last_line.append(round(tot,4))
#         writer.writerow(last_line)
#
#     print '\tloaded in >> %s.csv <<' %file_name

### check the correctness of start/end times of detections in path_file.txt
### and generate the relative path_file.csv file
def check_and_generate_csv(path_file):
    print 'check file >> %s.txt <<' %path_file
    file_input = open(path_file+'.txt').readlines()
    ### to count lines in file
    for i, l in enumerate(file_input):
        pass
    tot_detection = i-1

    list_detection = []
    error = False
    iter_detection = iter(file_input)
    try:
        ### two times next() to jump first and second line in file (header of tables)
        next(iter_detection)
        next(iter_detection)
        next_line = next(iter_detection)
        init_error = False
    except StopIteration:
        init_error = True
    if not init_error:
        for c in range(tot_detection):
            smart_line = []
            this_line = next_line.split('\t')
            for col in this_line:
                if col and not col.isspace():
                    col = col.strip()
                    smart_line.append(col)
            start = datetime.strptime(smart_line[0], '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(smart_line[1], '%Y-%m-%d %H:%M:%S')
            try:
                next_line = next(iter_detection)
                next_empty = False
            except StopIteration:
                next_empty = True
            if start <= end:
                # if not next_empty:
                #     ### check: this_line.end and next_line.start
                #     next_smart_line = []
                #     nxt_line = next_line.split('\t')
                #     for co in nxt_line:
                #         if co and not co.isspace():
                #             co = co.strip()
                #             next_smart_line.append(co)
                #     next_start = datetime.strptime(next_smart_line[0], '%Y-%m-%d %H:%M:%S')
                #     if end <= next_start:
                #         list_detection.append(smart_line)
                #     else:
                #         error = True
                #         print '\tstart/end time mismatch in lines %s and %s\n\t%s\n\t%s' %(counter+3, counter+4, smart_line, next_smart_line)
                #
                #     ### check correctness between start and start in next row
                #     next_smart_line = []
                #     nxt_line = next_line.split('\t')
                #     for co in nxt_line:
                #         if co and not co.isspace():
                #             co = co.strip()
                #             next_smart_line.append(co)
                #     next_start = datetime.strptime(next_smart_line[0], '%Y-%m-%d %H:%M:%S')
                #     if start <= next_start:
                #         list_detection.append(smart_line)
                #     else:
                #         error = True
                #         print '\tstart/end time mismatch in lines %s and %s\n\t%s\n\t%s' %(counter+3, counter+4, smart_line, next_smart_line)

                list_detection.append(smart_line) #comment if used above controls
            else:
                error = True
                print '\tstart/end time mismatch at line %s\n\t %s' %(c+3, smart_line)

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

### obtain probability of each adl
### return [list_adls] = list of adls and [p_adls]= list of their probability
### p(y) = p(adl)
def obtain_p_adls(path_file, house_name):
    tot_rows = len(open(path_file+'.csv').readlines())
    list_adls = []
    p_adls = []
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        for row in reader:
            adl = copy.deepcopy(row[2])
            exist_adl = False
            for c, a in enumerate(list_adls):
                if (a == adl):
                    p_adls[c] += 1
                    exist_adl = True
            if not exist_adl:
                list_adls.append(adl)
                p_adls.append(1)
        ### normalize
        partial_tot = 0
        for c, a in enumerate(list_adls):
            if c+1 == len(list_adls):
                p_adls[c] = round(1.00 - partial_tot, 4)
            else:
                p_adls[c] = round(np.divide(float(p_adls[c]), float(tot_rows)), 4)
                partial_tot += p_adls[c]
    print '%s: P(ADLs) calculated' %house_name
    csv_list(list_adls, p_adls, house_name+'_P(ADLs)')
    return list_adls, p_adls

### obtain probability of transition between adls
### return [t_adls] = [matrix] = matrix with transition probabilities
### p(y(t)|y(t-1)) = p(adl(t)|adl(t-1))
def obtain_t_adls(path_file, list_adls, house_name):
    ### initialize square matrix of transition between adls
    matrix = []
    for a in list_adls:
        l = []
        for b in list_adls:
            l.append(0)
        matrix.append(l)
    tot_rows = len(open(path_file+'.csv').readlines())
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        iter_file = iter(reader)
        next_row = next(iter_file)
        for i in range(tot_rows):
            this_a = next_row[2]
            try:
                next_row = next(iter_file)
                next_empty = False
            except StopIteration:
                next_empty = True
            if not next_empty:
                next_a = next_row[2]
                for x, a in enumerate(list_adls):
                    if a == this_a:
                        for y, b in enumerate(list_adls):
                            if b == next_a:
                                matrix[x][y] += 1

    normalize_matrix(matrix)
    print '%s : p(adls|adls) calculated' %house_name
    csv_square_matrix(list_adls, matrix, house_name+'_T(ADLs)')
    return matrix

def obtain_list_sens(path_file, house_name):
    ### build list of all possible triple Location-Type-Place
    list_t_sens = []
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        for row in reader:
            triple = '%s %s %s' %(copy.deepcopy(row[2]), copy.deepcopy(row[3]), copy.deepcopy(row[4]))
            exist_triple = False
            for t in list_t_sens:
                if t == triple:
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
    # csv_t_matrix(matrix, house_name+'_p(sens|adls)')
    del list_t_sens, matrix


def increment_sens_from_adls(matrix, t_sens, adl):
    for t in matrix:
        if t[0] == t_sens:
            for a in t[1]:
                if a[0] == adl:
                    a[1] += 1

def normalize_matrix(matrix):
    tot_in_row = []
    for z in enumerate(matrix):
        tot_in_row.append(0)
    for x, a in enumerate(matrix):
        for y, b in enumerate(matrix[x]):
            tot_in_row[x] += b
    for x, a in enumerate(matrix):
        if tot_in_row[x] != 0:
            part_in_row = 0
            for y, b in enumerate(matrix[x]):
                if y+1 == len(matrix):
                    matrix[x][y] = round(1.00 - part_in_row, 4)
                else:
                    matrix[x][y] = round(np.divide(float(matrix[x][y]), float(tot_in_row[x])), 4)
                    part_in_row += matrix[x][y]

# def p_adl(adl, house_name):
#     with open(house_name+'_p(adls).csv', 'rb') as csvfile:
#         reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
#         for a in reader:
#             if a[0] == adl:
#                 return a[1]

# def p_sen_from_adl(sen, adl, house_name):
#     tot = len(open(house_name+'_p(sens|adls).csv').readlines())
#     with open(house_name+'_p(sens|adls).csv', 'rb') as csvfile:
#         reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
#         index_a = 1
#         for count, s in enumerate(reader):
#             if count == 0:
#                 for c, a in enumerate(s):
#                     if a == adl:
#                         index_a = c
#             elif count != tot-1:
#                 if s[0] == sen:
#                     return s[index_a]
        # else:

# def p_adl_from_adl(adl1, adl2, house_name):
#     tot = len(open(house_name+'_p(adls|adls).csv').readlines())
#     with open(house_name+'_p(adls|adls).csv', 'rb') as csvfile:
#         reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
#         index_a1 = 1
#         index_a2 = 1
#         for count, act in enumerate(reader):
#             if count == 0:
#                 for c, a in enumerate(act):
#                     if a == adl2:
#                         index_a = c
#             elif count != tot-1:
#                 if act[0] == adl1:
#                     return act[index_a]
        # else:

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
        path_sens = house[3]

        temp = obtain_p_adls(path_adls, house_name)
        list_adls = temp[0]
        p_adls = temp[1]

        t_adls = obtain_t_adls(path_adls, list_adls, house_name)

        list_sens = obtain_list_sens(path_sens, house_name)
        # o_sens_adls = obtain_o_sens_adls(path_adls, list_adls, p_adls, path_sens, list_sens, house_name)

        # p_sens_from_adls(path_adls, list_adls, path_sens, list_sens, house_name)    ### calculate p(sens|adls)
        #
        # py = p_adl('Sleeping', house_name)
        # print 'p(Sleeping) = %s' %py
        #
        # pxy = p_sen_from_adl('Door PIR Bedroom', 'Sleeping', house_name)
        # print 'p(Door PIR Bedroom | Sleeping) = %s' %pxy
        #
        # pyy = p_adl_from_adl('Showering', 'Grooming', house_name)
        # print 'p(Grooming | Showering) = %s' %pyy
        # pyy = p_adl_from_adl('Grooming', 'Showering', house_name)
        # print 'p(Showering | Grooming) = %s' %pyy


        print '\n------------------------------------------------------------'

if __name__ == '__main__':
    project()
