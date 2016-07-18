

import csv
import copy
import numpy as np
from datetime import datetime

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

### normalize list structure
def normalize_list(list):
    norm_list = []
    epsilon = 0.0000001
    tot = 0
    part = 0
    for c, elem in enumerate(list):
        if list[c] == 0:
            list[c] = epsilon
    for elem in list:
        tot += elem
    for c, elem in enumerate(list):
        if c+1 == len(list):
            list[c] = 1.00 - part
        else:
            list[c] = np.divide(float(list[c]), float(tot))    
            part += list[c]

### print matrix structure
def print_matrix(matrix):
    for line in matrix:
        print line

### normalize matrix structure
def normalize_matrix(matrix):
    for row in matrix:
        normalize_list(row)

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

### create csv from matrix
def csv_matrix(list_rows, list_columns, matrix, file_name):
    with open(file_name +'.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
        first_line = ['\t']
        for x in list_columns:
            first_line.append(x)
        writer.writerow(first_line)
        new_line = []
        for c, y in enumerate(list_rows):
            new_line.append(y)
            for z in range(len(matrix)):
                new_line.append(matrix[c][z])
            writer.writerow(new_line)
            new_line = []
    print '\tloaded in >> %s.csv <<' %file_name

### obtain probability of each adl
### return [list_adls] = list of adls and [p_adls]= list of their probability
### p(y) = p(adl)
def obtain_p_adls(path_file, house_name):
    tot_rows = len(open(path_file+'.csv').readlines())
    list_adls = []
    p_adls = []
    seq_adls = []
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        for row in reader:
            adl = copy.deepcopy(row[2])
            exist_adl = False
            for c, a in enumerate(list_adls):
                if (a == adl):
                    p_adls[c] += 1
                    exist_adl = True
                    seq_adls.append(c)
            if not exist_adl:
                list_adls.append(adl)
                p_adls.append(1)
                seq_adls.append(len(list_adls)-1)
        normalize_list(p_adls)
    print '%s: P(ADLs) calculated' %house_name
    csv_list(list_adls, p_adls, house_name+'_P(ADLs)')
    return list_adls, p_adls, seq_adls

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
    print '%s : T(ADLs) calculated' %house_name
    csv_matrix(list_adls, list_adls, matrix, house_name+'_T(ADLs)')
    return matrix

def obtain_list_sens(path_file, house_name):
    ### build list of all possible triple Location-Type-Place
    ### ad save the order of sensors detections
    list_sens = []
    seq_sens = []
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        for row in reader:
            sens = '%s %s %s' %(copy.deepcopy(row[2]), copy.deepcopy(row[3]), copy.deepcopy(row[4]))
            exist_sens = False
            for c, s in enumerate(list_sens):
                if s == sens:
                    exist_sens = True
                    seq_sens.append(c)
            if not exist_sens:
                list_sens.append(sens)
                seq_sens.append(len(list_sens)-1)
    return list_sens, seq_sens

###
def verbose_sequence_to_int(list_name, verb_sequence):
    int_sequence = []
    for a in verb_sequence:
        for x, b in enumerate(list_name):
            if a == b:
                int_sequence.append(x)
    return int_sequence

###
def int_sequence_to_verbose(list_name, int_sequence):
    verb_sequence = []
    for a in int_sequence:
        for x, b in enumerate(list_name):
            if a == x:
                verb_sequence.append(b)
    return verb_sequence

### obtain probability of observations sens from adls
### return [o_sens_adls] = [matrix] = matrix with observations probabilities
### p(x|y) = p(sens|adls)
def obtain_o_sens_adls(path_adls, list_adls, path_sens, list_sens, house_name):
    ### initialize matrix of observation 
    matrix = []
    for a in list_adls:
        l = []
        for s in list_sens:
            l.append(0)
        matrix.append(l)

    with open(path_adls+'.csv', 'rb') as csv_adls:
        reader_adls = csv.reader(csv_adls, dialect='excel', delimiter='\t')
        for a in reader_adls:
            adl = a[2]
            start_a = datetime.strptime(a[0], '%Y-%m-%d %H:%M:%S')
            end_a = datetime.strptime(a[1], '%Y-%m-%d %H:%M:%S')
            for ca, aa in enumerate(list_adls):
                if aa == adl:
                    break
            # print '%s_%s: %s - %s' %(ca, adl, start_a, end_a)
            with open(path_sens+'.csv', 'rb') as csv_sens:
                reader_sens = csv.reader(csv_sens, dialect='excel', delimiter='\t')
                for s in reader_sens:
                    sens = '%s %s %s' %(s[2],s[3],s[4])
                    start_s = datetime.strptime(s[0], '%Y-%m-%d %H:%M:%S')
                    end_s = datetime.strptime(s[1], '%Y-%m-%d %H:%M:%S')
                    for cs, ss in enumerate(list_sens):
                        if ss == sens:
                            break
                    if (start_s >= start_a) and (end_s <= end_a):
                        matrix[ca][cs] += 1
                        # print '\t%s_%s: %s - %s' %(cs, sens, start_s, end_s)
    normalize_matrix(matrix)
    # print_matrix(matrix)
    print '%s : O(Sens|ADLs) calculated' %house_name
    csv_matrix(list_adls, list_sens, matrix, house_name+'_O(Sens|ADLs)')
    return matrix

def remove_repetitions(seq):
    new_seq = []
    for x, s in enumerate(seq):
        if x == 0:
            new_seq.append(seq[x])
        else:
            if seq[x] != seq[x-1]:
                new_seq.append(seq[x])
    return new_seq

def diff_base(seq1, seq2):
    n_diffs = 0
    if len(seq1) < len(seq2):
        l = len(seq1)-1
    else:
        l = len(seq2)-1
    for x in range(l):
        mismatch = ( seq1[x] != seq2[x] )
        if mismatch:
            n_diffs += 1
        print '%s \t %s \t:%s' %(seq1[x], seq2[x], not mismatch)
    print 'n total diffs: %s' %n_diffs
    return n_diffs

def compare_A(seq1, seq2, x, p_diff):
    
    if seq1[x] == seq2[x]:
        p_diff += 1.0
        return p_diff
    
    try:
        if seq1[x] == seq2[x-1]:
            p_diff += 0.8
            return p_diff
    except IndexError:
        pass
    try:
        if seq1[x] == seq2[x+1]:
            p_diff += 0.8
            return p_diff
    except IndexError:
        pass
    try:
        if seq1[x] == seq2[x-2]:
            p_diff += 0.6
            return p_diff
    except IndexError:
        pass
    try:
        if seq1[x] == seq2[x+2]:
            p_diff += 0.6
            return p_diff
    except IndexError:
        pass
    try:
        if seq1[x] == seq2[x-3]:
            p_diff += 0.4
            return p_diff
    except IndexError:
        pass
    try:
        if seq1[x] == seq2[x+3]:
            p_diff += 0.4
            return p_diff
    except IndexError:
        pass
    
    p_diff += 0.001
    return p_diff

def diff_A(seq1, seq2):
    p_diff = 0
    last = False
    iter_seq1 = iter(seq1)
    for x in range(len(seq1)):
        try:
            k = next(iter_seq1)
            last = False
        except StopIteration:
            last = True
        if not last:
            p_diff = compare_A(seq1, seq2, x, p_diff)
    p_diff = np.divide(p_diff, float(len(seq1)))            
    return p_diff

def compare_B(seq1, seq2, x, y):
    
    if seq1[x] == seq2[y]:
        print '== seq[%s] %s == seq[%s] %s -> return %s' %(x, seq1[x], y, seq2[y], y+1)
        return y+1
    
    try:
        if seq1[x] == seq2[y+1]:
            print '+1 seq[%s] %s == seq[%s] %s -> return %s' %(x, seq1[x], y+1, seq2[y+1], y+2)
            return y+2
    except IndexError:
        pass
    try:
        if seq1[x] == seq2[y+2]:
            print '+2 seq[%s] %s == seq[%s] %s -> return %s' %(x, seq1[x], y+2, seq2[y+2], y+3)
            return y+3
    except IndexError:
        pass
    try:
        if seq1[x] == seq2[y+3]:
            print '+3 seq[%s] %s == seq[%s] %s -> return %s' %(x, seq1[x], y+3, seq2[y+3], y+4)
            return y+4
    except IndexError:
        pass
    
    return y

def diff_B(seq1, seq2):
    p_diff = 0
    y = 0
    iter_seq1 = iter(seq1)
    for x in range(len(seq1)):
        try:
            k = next(iter_seq1)
            last = False
        except StopIteration:
            last = True
        if not last:
            if y <= len(seq2):
                y = compare_B(seq1, seq2, x, y)
    print y
             

            
             

def elaborate_dataset():
    print 
    for path_file in house:
        if ('ADLs' in path_file) or ('Sensors' in path_file):
            check_and_generate_csv(path_file)

    house_name = house[0]
    path_adls = house[2]
    path_sens = house[3]

    temp = obtain_p_adls(path_adls, house_name)
    list_adls = temp[0]
    p_adls = temp[1]
    seq_adls = temp[2]
    t_adls = obtain_t_adls(path_adls, list_adls, house_name)

    temp = obtain_list_sens(path_sens, house_name)
    list_sens = temp[0]
    seq_sens = temp[1]
    o_sens_adls = obtain_o_sens_adls(path_adls, list_adls, path_sens, list_sens, house_name)

    return house_name, path_adls, list_adls, p_adls, seq_adls, t_adls, path_sens, list_sens, seq_sens, o_sens_adls


def model_hmm(package, house_name, path_adls, list_adls, p_adls, seq_adls, t_adls, path_sens, list_sens, seq_sens, o_sens_adls):

    ### pomegranate
    if package == 'pomegranate':

        model = HiddenMarkovModel( name="Smarthome" )

        states = []
        for x, a in enumerate(o_sens_adls):
            st = {}
            for y, s in enumerate(a):
                st['%s' %list_sens[y]] = o_sens_adls[x][y]
            state = State( DiscreteDistribution( st ), name='%s' %list_adls[x] )
            states.append(state)
            model.add_transition( model.start, state, p_adls[x] )

        for x, a in enumerate(t_adls):
            for y, aa in enumerate(a):
                model.add_transition( states[x], states[y], t_adls[x][y] )

        # for state in states:
            # print state

        model.bake()

        sequence = int_sequence_to_verbose(list_sens, seq_sens)

        full_viterbi_adls = (' '.join( state.name for i, state in model.viterbi( sequence )[1] )).split( )
        viterbi_adls = remove_repetitions( verbose_sequence_to_int( list_adls, full_viterbi_adls ) )
        print viterbi_adls
        print seq_adls

        # diff_base(seq_adls, viterbi_adls)
        diff_A(seq_adls, viterbi_adls)
        diff_B(seq_adls, viterbi_adls)

        # print model.predict(sequence, algorithm='viterbi')
        # print
        # print "\n".join( state.name for i, state in model.viterbi( sequence )[1] )
        # int_seq_gen = verbose_sequence_to_int(list_adls, verb_seq_gen)




    ### ghmm
    if package == 'ghmm':

        sigma = IntegerRange(0, len(list_sens)) ### emission domain
    
        model = HMMFromMatrices(sigma, DiscreteDistribution(sigma), t_adls, o_sens_adls, p_adls)
        # print '\n', model

        sequence = EmissionSequence(e, seq_sens)

        vt = model.viterbi(sequence)
        print 'analyzed >test_seq< with viterbi algorithm\n', vt


if __name__ == '__main__':
    
    ### dataset is the list of each house analyzed, in each house:
    ###     house[0] = name house
    ###     house[1] = Description
    ###     house[2] = ADLs
    ###     house[3] = Sensors

    ordonezA = ['Dataset/OrdonezA','Dataset/OrdonezA_Description','Dataset/OrdonezA_ADLs','Dataset/OrdonezA_Sensors']
    # ordonezA = ['Dataset/OrdonezB','Dataset/OrdonezB_Description','Dataset/OrdonezB_ADLs','Dataset/OrdonezB_Sensors']
    # dataset = [ordonezA, ordonezB]
    # dataset = [ordonezA]

    test1 = ['Deposet/test1','Deposet/test1_Description','Deposet/test1_ADLs','Deposet/test1_Sensors']
    test2 = ['Deposet/test2','Deposet/test2_Description','Deposet/test2_ADLs','Deposet/test2_Sensors']
    dataset = [test1]

    ### packages is the list of package that can be used to build hmm model
    # packages = ['pomegranate', 'ghmm']
    packages = ['pomegranate']

    for house in dataset:

        data = elaborate_dataset()
        
        house = data[0]
        path_adls = data[1]
        list_adls = data[2]
        p_adls = data[3]
        seq_adls = data[4]
        t_adls = data[5]
        path_sens = data[6]
        list_sens = data[7]
        seq_sens = data[8]
        o_sens_adls = data[9]

        for pack in packages:

            if pack == 'ghmm':
                from ghmm import *
            
            if pack == 'pomegranate':
                from pomegranate import *
            
            model_hmm(pack, house, path_adls, list_adls, p_adls, seq_adls, t_adls, path_sens, list_sens, seq_sens, o_sens_adls)
