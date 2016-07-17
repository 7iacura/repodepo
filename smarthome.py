

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
        normalize_list(p_adls)
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
    print '%s : T(ADLs) calculated' %house_name
    csv_matrix(list_adls, list_adls, matrix, house_name+'_T(ADLs)')
    return matrix

def obtain_list_sens(path_file, house_name):
    ### build list of all possible triple Location-Type-Place
    ### ad save the order of sensors detections
    list_sens = []
    sens_seq = []
    with open(path_file+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, dialect='excel', delimiter='\t')
        for row in reader:
            sens = '%s %s %s' %(copy.deepcopy(row[2]), copy.deepcopy(row[3]), copy.deepcopy(row[4]))
            exist_sens = False
            for c, s in enumerate(list_sens):
                if s == sens:
                    exist_sens = True
                    sens_seq.append(c)
            if not exist_sens:
                list_sens.append(sens)
                sens_seq.append(len(list_sens)-1)
    return list_sens, sens_seq

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

def elaborate_dataset():

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
    dataset = [test1]

    for house in dataset:
        print 
        # for path_file in house:
            ### check the correctness of file
            # if ('ADLs' in path_file) or ('Sensors' in path_file):
                # check_and_generate_csv(path_file)

        house_name = house[0]
        path_adls = house[2]
        path_sens = house[3]

        temp = obtain_p_adls(path_adls, house_name)
        list_adls = temp[0]
        p_adls = temp[1]
        t_adls = obtain_t_adls(path_adls, list_adls, house_name)

        temp = obtain_list_sens(path_sens, house_name)
        list_sens = temp[0]
        sens_seq = temp[1]
        o_sens_adls = obtain_o_sens_adls(path_adls, list_adls, path_sens, list_sens, house_name)

        return house_name, path_adls, list_adls, p_adls, t_adls, path_sens, list_sens, sens_seq, o_sens_adls

def ghmm(sigma, A, B, pi, train_seq, test_seq):
    m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)
    print '\n', m

    m.baumWelch(train_seq)
    print 'trained with baumWelch method\n'

    vt = m.viterbi(test_seq)
    print 'analyzed >test_seq< with viterbi algorithm\n', vt

    # my_seq = EmissionSequence(sigma, [1] * 20 + [6] * 10 + [1] * 40)
    # vm = m.viterbi(my_seq)
    # print
    # print 'analyzed >my_seq< with viterbi algorithm\n', vm

    return m

def model_hmm(package, house_name, path_adls, list_adls, p_adls, t_adls, path_sens, list_sens, sens_seq, o_sens_adls):

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

        # model.bake( verbose=True )
        model.bake()

        sequence = []
        for ss in sens_seq:
            for y, ls in enumerate(list_sens):
                if ss == y:
                    sequence.append(ls)

        # model.fit(sequence, algorithm='baum-welch')

        # print model.predict(sequence, algorithm='viterbi')

        # print "\n".join( state.name for i, state in model.viterbi( sequence )[1] )

        sample = verbose_sequence_to_int(list_sens, model.sample(30))




        # rainy = State( DiscreteDistribution({ 'walk': 0.1, 'shop': 0.4, 'clean': 0.5 }), name='Rainy' )
        # sunny = State( DiscreteDistribution({ 'walk': 0.6, 'shop': 0.3, 'clean': 0.1 }), name='Sunny' )

        # model.add_transition( model.start, rainy, 0.6 )
        # model.add_transition( model.start, sunny, 0.4 )
        
        # model.add_transition( rainy, rainy, 0.65 )
        # model.add_transition( rainy, sunny, 0.25 )
        # model.add_transition( sunny, rainy, 0.35 )
        # model.add_transition( sunny, sunny, 0.55 )
        
        # model.add_transition( rainy, model.end, 0.1 )
        # model.add_transition( sunny, model.end, 0.1 )
        # model.bake( verbose=True )
            
        # sequence = [ 'walk', 'shop', 'clean', 'clean', 'clean', 'walk', 'clean' ]
        
        # print math.e**model.forward( sequence )[ len(sequence), model.end_index ]
        # print math.e**model.forward_backward( sequence )[1][ 2, model.states.index( rainy ) ]
        # print math.e**model.backward( sequence )[ 3, model.states.index( sunny ) ]
        # print " ".join( state.name for i, state in model.maximum_a_posteriori( sequence )[1] )


    if package == 'ghmm':

        e = IntegerRange(0, len(list_sens)) ### emission domain
        # seq_for_training = [1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 9, 10, 3, 11, 3, 4, 12, 12, 6, 11, 4]
        train = EmissionSequence(e, sens_seq)
        ts = [0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 4, 9, 2, 10, 2, 3, 11, 11, 5, 10, 3]
        test = EmissionSequence(e, ts)
    
        model = ghmm(e, t_adls, o_sens_adls, p_adls, train, test)

        ### //////////////////////////////////////////////
        ### example from ghmm.org
        # print '\n//////////////////////////////////////////////\nexample from ghmm.org'
        # sigma = IntegerRange(1, 7)
        # A = [[0.9, 0.1], [0.3, 0.7]]
        # efair = [1.0 / 6] * 6
        # eloaded = [3.0 / 13, 3.0 / 13, 2.0 / 13, 2.0 / 13, 2.0 / 13, 1.0 / 13]
        # B = [efair, eloaded]
        # pi = [0.5] * 2
        # tr = [1, 6, 2, 5, 5, 4, 5, 1, 2, 1, 3, 6, 6, 3, 2, 1, 4, 4, 1, 1, 4, 2, 1, 1, 6, 3, 3, 2, 1, 4, 4, 3, 3, 5, 3, 3, 3, 3, 4, 3, 1, 5, 4, 1, 4, 5, 1, 1, 3, 4, 3, 5, 5, 1, 5, 2, 1, 5, 3, 6, 3, 6, 5, 6, 5, 3, 3, 1, 2, 6, 3, 3, 2, 2, 5, 4, 1, 5, 6, 3, 3, 5, 1, 5, 2, 3, 1, 1, 1, 5, 6, 4, 5, 5, 1, 6, 2, 6, 5, 3, 1, 1, 3, 3, 1, 1, 2, 5, 2, 3, 2, 4, 1, 5, 5, 5, 4, 6, 5, 6, 3, 1, 6, 1, 5, 4, 3, 1, 3, 1, 2, 6, 3, 5, 2, 1, 3, 6, 4, 4, 4, 4, 3, 5, 1, 6, 4, 4, 3, 5, 1, 5, 5, 5, 5, 6, 5, 1, 6, 1, 1, 4, 1, 4, 2, 6, 6, 2, 5, 4, 5, 5, 4, 3, 3, 6, 5, 4, 1, 5, 3, 3, 3, 2, 5, 6, 2, 3, 3, 5, 1, 3, 4, 5, 1, 6, 3, 2, 4, 5, 2, 2, 5, 1, 4, 5, 1, 5, 6, 5, 3, 4, 6, 1, 3, 2, 4, 6, 6, 3, 1, 6, 5, 2, 5, 4, 4, 4, 4, 2, 6, 3, 3, 2, 1, 1, 6, 5, 3, 3, 3, 4, 5, 6, 1, 6, 2, 3, 1, 4, 6, 5, 3, 1, 2, 4, 6, 2, 6, 2, 1, 2, 6, 5, 6, 1, 4, 4, 1, 5, 5, 5, 3, 4, 4, 5, 2, 2, 5, 1, 2, 1, 3, 2, 3, 3, 3, 5, 3, 5, 2, 3, 5, 5, 5, 2, 6, 2, 5, 6, 4, 5, 4, 4, 3, 3, 6, 2, 3, 2, 2, 6, 1, 1, 1, 4, 1, 2, 6, 1, 4, 5, 2, 6, 2, 6, 6, 3, 2, 4, 2, 2, 2, 3, 3, 4, 4, 1, 1, 1, 5, 3, 4, 3, 5, 3, 3, 3, 4, 6, 1, 1, 6, 4, 2, 4, 6, 4, 1, 5, 1, 4, 5, 5, 4, 6, 1, 3, 5, 1, 5, 3, 1, 6, 6, 3, 3, 2, 4, 4, 3, 1, 2, 5, 3, 5, 4, 5, 3, 2, 2, 6, 4, 4, 2, 6, 4, 6, 1, 4, 6, 3, 3, 4, 6, 3, 2, 3, 5, 3, 5, 4, 2, 2, 2, 1, 4, 1, 5, 1, 1, 1, 6, 6, 3, 4, 4, 5, 1, 2, 4, 3, 3, 1, 1, 5, 6, 1, 3, 4, 6, 2, 4, 5, 2, 1, 1, 2, 6, 5, 4, 3, 5, 4, 5, 2, 5, 1, 1, 5, 4, 3, 6, 5, 3, 6, 4, 2, 3, 5, 1, 4, 6, 1, 2, 6, 1, 3, 5, 2, 1, 2, 6, 1, 2, 5, 4, 4, 5, 5, 1, 4, 3, 1, 6, 5, 4, 3, 4, 2, 4, 2, 6, 6, 3, 3, 6, 1, 5, 6, 4, 1, 3, 2, 2, 1, 3, 4, 6, 3, 5, 5, 1, 5, 5, 2, 5, 3, 3, 1, 3, 4, 2, 1, 1, 2, 1, 5, 1, 5, 4, 2, 6, 3, 5, 6, 6, 2, 2, 5, 6, 1, 5, 3, 1, 2, 1, 2, 3, 4, 5, 5, 4, 3, 3, 3, 4, 6, 2, 5, 2, 5, 5, 3, 1, 4, 3, 6, 2, 5, 6, 3, 4, 2, 6, 2, 5, 1, 3, 1, 2, 2, 1, 4, 1, 6, 1, 4, 5, 5, 5, 2, 3, 2, 5, 1, 6, 2, 5, 5, 1, 4, 4, 6, 1, 3, 1, 1, 4, 3, 1, 2, 1, 1, 1, 4, 2, 2, 2, 1, 2, 3, 5, 5, 4, 4, 4, 6, 3, 6, 4, 3, 5, 2, 4, 4, 5, 6, 2, 6, 5, 5, 3, 1, 5, 4, 4, 1, 5, 3, 3, 1, 2, 6, 3, 6, 6, 6, 6, 1, 1, 3, 2, 5, 6, 4, 3, 4, 6, 5, 2, 2, 1, 2, 5, 5, 6, 1, 5, 2, 5, 6, 2, 4, 5, 4, 2, 5, 5, 4, 3, 4, 1, 6, 1, 1, 6, 2, 2, 1, 3, 4, 3, 1, 1, 6, 1, 1, 1, 4, 4, 4, 1, 5, 6, 4, 4, 2, 1, 4, 6, 6, 2, 3, 6, 2, 4, 3, 4, 3, 3, 3, 4, 1, 2, 4, 3, 3, 6, 2, 4, 1, 1, 2, 5, 4, 6, 3, 1, 2, 1, 3, 2, 1, 3, 6, 2, 2, 2, 5, 1, 1, 6, 6, 6, 1, 4, 2, 6, 2, 3, 3, 2, 4, 1, 2, 5, 2, 6, 3, 5, 6, 2, 1, 3, 6, 5, 1, 4, 1, 4, 6, 4, 4, 4, 2, 2, 3, 6, 4, 6, 5, 4, 2, 2, 5, 5, 3, 4, 2, 3, 3, 6, 6, 3, 1, 5, 5, 5, 6, 4, 2, 6, 5, 4, 5, 6, 3, 1, 1, 5, 4, 5, 5, 3, 5, 3, 2, 2, 1, 3, 3, 6, 5, 3, 5, 1, 2, 5, 2, 4, 1, 5, 2, 1, 2, 5, 3, 1, 3, 4, 4, 1, 5, 1, 3, 1, 2, 6, 3, 6, 1, 3, 5, 2, 6, 6, 5, 2, 3, 6, 3, 3, 4, 5, 1, 6, 4, 2, 4, 1, 4, 3, 3, 4, 2, 3, 1, 6, 2, 4, 3, 5, 2, 2, 4, 5, 1, 3, 2, 2, 1, 6, 3, 2, 1, 2, 1, 5, 4, 2, 1, 4, 5, 5, 3, 4, 4, 6, 5, 3, 6, 5, 3, 6, 5, 6, 3, 3, 5, 3, 3, 2, 4, 2, 6, 3, 2, 6, 5, 5, 5, 3, 3, 1, 2, 5, 1, 3, 6, 3, 3, 5, 1, 4, 4, 4, 2, 3, 4, 2, 6, 1, 6, 1, 4, 3, 2, 1, 3, 5, 4, 6, 5, 6, 6, 6, 4, 2, 4, 3, 5, 1, 1, 1, 5, 4, 6, 2, 2, 4, 3, 2, 1, 6, 2, 1, 6, 1, 6, 5, 1, 6, 4, 3, 6, 5, 4, 4, 1, 3, 4, 3, 1, 6, 4, 4, 2, 1, 2, 5, 5, 1, 4, 3, 2, 3, 2, 1, 4, 6, 3, 6, 6, 1, 5, 3, 6, 1, 3, 6, 6, 5, 3, 2, 4, 6, 2, 4, 4, 4, 2, 4, 4, 3, 6, 5, 3, 4, 1, 3, 3, 1, 6, 1, 4, 1, 6, 3, 5, 2, 1, 5, 4, 2, 4, 6, 1, 4, 3, 1, 1, 3, 2, 5, 5, 1, 1, 5, 3, 4, 3, 1, 3, 2, 5, 6, 2, 1, 3, 2, 6, 3, 6, 4, 4, 3, 5, 3, 2, 5, 2, 3, 2, 4, 2, 6, 2, 4, 6, 5, 5, 2, 3, 5, 6, 4, 3, 1, 3, 3, 2, 2, 2, 3, 6, 1, 6, 3, 1, 6, 3, 1, 3, 1, 1, 1, 4, 3, 1, 5, 4, 6, 2, 1, 6, 2, 2, 2, 1, 5, 5, 1, 2, 5, 5, 2, 5, 2, 4, 5, 1, 5, 5, 6, 6, 5, 3, 5, 6, 2, 5, 5, 5, 1, 2, 2, 4, 2, 4, 5, 5, 1, 4, 1, 5, 3, 5, 1, 4, 2, 1, 2, 2, 2, 4, 4, 4, 4, 2, 1, 1, 4, 5, 4, 1, 2, 2, 3, 5, 6, 4, 1, 1, 1, 3, 6, 1, 2, 4, 3, 2, 3, 2, 3, 6, 6, 3, 4, 4, 4, 6, 6, 2, 6, 6, 3, 2, 3, 5, 1, 2, 4, 1, 3, 5, 5, 1, 2, 5, 1, 5, 6, 2, 6, 2, 1, 1, 4, 4, 2, 1, 5, 2, 3, 4, 4, 2, 5, 2, 5, 5, 5, 3, 1, 4, 6, 6, 5, 5, 1, 3, 3, 6, 5, 6, 2, 1, 1, 1, 5, 4, 3, 1, 1, 2, 1, 3, 1, 6, 1, 5, 1, 2, 6, 1, 2, 2, 2, 5, 1, 2, 6, 5, 2, 2, 1, 3, 6, 2, 6, 1, 1, 6, 3, 5, 2, 6, 3, 1, 1, 4, 3, 5, 2, 3, 2, 4, 5, 1, 2, 5, 5, 3, 1, 4, 4, 5, 5, 5, 3, 2, 1, 3, 1, 3, 2, 2, 5, 6, 5, 2, 2, 6, 4, 1, 4, 2, 3, 5, 2, 5, 5, 6, 4, 6, 4, 3, 5, 4, 3, 5, 2, 6, 5, 5, 5, 1, 4, 6, 5, 3, 5, 5, 4, 5, 4, 4, 3, 1, 6, 4, 1, 5, 5, 6, 3, 2, 4, 3, 4, 4, 4, 5, 3, 3, 4, 2, 5, 3, 6, 1, 4, 5, 5, 6, 1, 4, 1, 1, 2, 2, 1, 3, 5, 1, 1, 2, 1, 2, 1, 1, 2, 3, 1, 3, 5, 5, 6, 4, 3, 5, 6, 1, 3, 4, 5, 6, 3, 1, 5, 4, 2, 3, 2, 6, 4, 6, 6, 6, 4, 1, 5, 1, 6, 4, 6, 6, 1, 3, 1, 4, 6, 2, 1, 4, 2, 4, 4, 4, 6, 6, 1, 6, 3, 6, 5, 3, 5, 3, 5, 6, 6, 1, 6, 2, 2, 1, 6, 1, 1, 1, 6, 3, 3, 6, 5, 1, 1, 4, 4, 2, 6, 5, 3, 4, 6, 5, 4, 2, 2, 1, 5, 3, 5, 5, 5, 4, 4, 6, 4, 4, 3, 1, 1, 6, 2, 3, 3, 3, 6, 5, 2, 1, 3, 1, 4, 4, 2, 3, 1, 2, 6, 1, 6, 4, 4, 3, 1, 4, 6, 1, 5, 4, 5, 3, 2, 2, 5, 5, 3, 6, 1, 2, 5, 5, 5, 4, 4, 4, 5, 1, 2, 4, 1, 6, 4, 3, 4, 4, 2, 4, 3, 4, 3, 5, 2, 3, 2, 3, 6, 6, 5, 2, 5, 5, 1, 4, 3, 5, 6, 5, 5, 1, 5, 3, 1, 3, 2, 2, 5, 6, 2, 6, 6, 4, 5, 2, 3, 1, 5, 5, 2, 2, 6, 5, 2, 5, 5, 3, 4, 6, 2, 1, 5, 6, 3, 2, 3, 2, 6, 6, 1, 4, 6, 5, 6, 6, 6, 5, 6, 6, 3, 6, 6, 4, 5, 5, 4, 6, 5, 2, 6, 6, 3, 3, 3, 3, 6, 5, 5, 6, 1, 3, 6, 1, 1, 1, 3, 4, 4, 4, 1, 4, 6, 6, 1, 5, 4, 1, 3, 2, 4, 5, 3, 6, 2, 3, 3, 3, 4, 5, 3, 2, 4, 4, 2, 3, 6, 4, 6, 6, 1, 2, 1, 2, 3, 1, 6, 5, 4, 4, 4, 5, 6, 6, 1, 5, 4, 1, 4, 3, 2, 1, 3, 2, 3, 1, 6, 5, 2, 3, 6, 6, 1, 1, 5, 2, 2, 1, 3, 2, 5, 2, 6, 5, 4, 5, 2, 2, 1, 6, 3, 1, 5, 1, 6, 4, 6, 3, 3, 1, 1, 4, 4, 6, 4, 2, 1, 6, 2, 4, 3, 4, 4, 5, 5, 2, 1, 4, 3, 5, 6, 6, 6, 2, 5, 6, 2, 3, 2, 4, 5, 6, 3, 3, 5, 4, 1, 1, 1, 2, 5, 6, 1, 4, 5, 3, 4, 2, 1, 6, 6, 4, 2, 6, 4, 3, 1, 2, 4, 2, 4, 6, 5, 6, 5, 6, 4, 4, 1, 2, 4, 6, 6, 1, 5, 2, 3, 2, 1, 5, 4, 1, 4, 4, 3, 6, 4, 6, 1, 6, 6, 2, 3, 5, 4, 3, 2, 1, 4, 3, 2, 2, 4, 5, 2, 1, 6, 1, 6, 1, 5, 1, 3, 1, 1, 6, 5, 6, 6, 6, 4, 6, 3, 4, 1, 4, 1, 1, 6, 3, 1, 3, 3, 4, 5, 2, 5, 4, 3, 1, 5, 5, 6, 6, 6, 5, 3, 6, 2, 4, 5, 6, 2, 3, 2, 3, 1, 3, 6, 4, 4, 5, 5, 5, 3, 1, 1, 6, 1, 4, 1, 2, 3, 3, 2, 3, 4, 1]
        # train_seq = EmissionSequence(sigma, tr)
        # test_seq = EmissionSequence(sigma, [4, 6, 1, 1, 5, 2, 5, 3, 4, 2, 1, 6, 1, 5, 5, 5, 6, 2, 4, 3, 4, 3, 4, 1, 3, 4, 2, 2, 3, 3, 2, 6, 6, 3, 6, 4, 1, 4, 4, 4, 6, 2, 1, 1, 2, 2, 2, 3, 5, 1, 2, 1, 4, 2, 6, 1, 6, 4, 4, 1, 1, 4, 6, 5, 1, 2, 5, 6, 3, 5, 1, 1, 2, 2, 1, 1, 5, 4, 6, 6, 3, 5, 4, 4, 4, 3, 3, 6, 6, 2, 1, 2, 1, 3, 2, 6, 2, 4, 2, 4])
        # test_vit = []

        # m = ghmm_matrix(sigma, A, B, pi, train_seq, test_seq)
        ### //////////////////////////////////////////////



if __name__ == '__main__':
    
    # packages = ['ghmm', 'pomegranate']
    # packages = ['ghmm']
    packages = ['pomegranate']

    data = elaborate_dataset()
    house_name = data[0]
    path_adls = data[1]
    list_adls = data[2]
    p_adls = data[3]
    t_adls = data[4]
    path_sens = data[5]
    list_sens = data[6]
    sens_seq = data[7]
    o_sens_adls = data[8]

    for package in packages:

        if package == 'ghmm':
            from ghmm import *
        
        if package == 'pomegranate':
            from pomegranate import *
            import random
            import math
        
        model_hmm(package, house_name, path_adls, list_adls, p_adls, t_adls, path_sens, list_sens, sens_seq, o_sens_adls)
