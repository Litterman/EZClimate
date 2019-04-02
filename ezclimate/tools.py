# -*- coding: utf-8 -*-
import numpy as np
import csv

###########
### I/O ###
###########

def find_path(file_name, directory="data", file_type=".csv"):
    import os
    cwd = os.getcwd()
    if not os.path.exists(directory):
        os.makedirs(directory)
    d = os.path.join(cwd, os.path.join(directory,file_name+file_type))
    return d

def create_file(file_name):
    import os
    d = find_path(file_name)
    if not os.path.isfile(d):
        open(d, 'w').close()
    return d

def file_exists(file_name):
    import os
    d = find_path(file_name)
    return os.path.isfile(d)

def load_csv(file_name, delimiter=';', comment=None):
    d = find_path(file_name)
    pass

def write_columns_csv(lst, file_name, header=[], index=None, start_char=None, delimiter=';', open_as='w'):
    """
    write_columns_csv outputs tree data to an NEW (not existing) csv file

    lst       : a list of a list containing data for a single tree
    file_name : 
    headers   : names of the trees; these are put in the first row of the csv file.
    index     : index data (e.g., Year and Node)  
                  - NB: Header should have the index names as the first element(s)
    
    """
    d = find_path(file_name)
    if file_name.find('tree') >0:
        print('***in write_column_csv, file_name =',file_name,' header =',header,'index=',index)
        print('***lst = ',lst)    
    if index is not None:
        index.extend(lst)
        output_lst = list(zip(*index))
    else:
        output_lst = list(zip(*lst))
    if file_name.find('tree') >0:
        print()
        print('***in write_column_csv,output_lst=',output_lst)
        print()

    with open(d, open_as) as f:
        writer = csv.writer(f, delimiter=delimiter)
        if start_char is not None:
            writer.writerow([start_char])
        if header:
            writer.writerow(header)
        for row in output_lst:
            if file_name.find('tree') >0:
                print('    ***write_columns_csv -- type:',type(row),', row=', row)
            writer.writerow(row)
        if file_name.find('tree') >0:
            print('***DONE -- rite_columns_csv')
            print()
    if file_name.find('tree') >0:
        x = input('WCC-- Please halt the program here and examine the _trees file in the data folder') 
        

def write_columns_to_existing(lst, file_name, header="", delimiter=';'):
    d = find_path(file_name)
    #print('***In WCTE, len(lst) shape = ',len(lst))
    with open(d, 'r') as finput:
        reader = csv.reader(finput, delimiter=delimiter)
        all_lst = []
        row = next(reader)
        #print('***In WCTE, row- = ', row)
        nested_list = isinstance(lst[0], list) or isinstance(lst[0], np.ndarray)
        #print('***In WCTE, type(lst[0]) ',type(lst[0]))
        if nested_list:
            lst = list(zip(*lst))
            row.extend(header)    
        else:
            row.append(header)
        #print('***In WCTE, after header, row = ', row)
        #print('***In WCTE, after header, len(lst) = ',len(lst))

        all_lst.append(row)
        n = len(lst)
        i = 0
        for row in reader:
            #print(' *** In WCTE i=',i,', and row =',row)
            if i >= (n+10):
                print('***(temporary) emergency break with i=',i)
                break
            #print('***In WCTE, row ',i,'- = ', row)
            #print('***IN WCTE, len(lst),i,lst[i]',len(lst),i,lst[i])
            if nested_list:
                row.extend(lst[i])
            else:
                try:
                    item = lst[i]
                except IndexError:
                    print(' *** Index Error (1) in WCTE; i=',i)
                    print(' *** Index Error (1) in WCTE; len(lst)=',len(lst))
                    print(' *** Index Error (2) in WCTE; row=', row)
                try:
                    row.append(item)
                except IndexError:
                    print(' *** Index Error (2) in WCTE; i=',i)

            all_lst.append(row)
            i += 1
            
    with open(d, 'w') as foutput:
        writer = csv.writer(foutput, delimiter=delimiter)
        writer.writerows(all_lst)
    if file_name.find('tree') >0:
        x = input('WCTE - Please halt the program here and examine the _trees file in the data folder') 

            
def append_to_existing(lst, file_name, header="", index=None, delimiter=';', start_char=None):
    write_columns_csv(lst, file_name, header, index, start_char=start_char, delimiter=delimiter, open_as='a')

def import_csv(file_name, delimiter=';', header=True, indices=None, start_at=0, break_at='\n', ignore=""):
    d = find_path(file_name)
    input_lst = []
    indices_lst = []
    with open(d, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in range(0, start_at):
            next(reader)
        if header:
            header_row = next(reader)
        for row in reader:
            if row[0] == break_at:
                break
            if row[0] == ignore:
                continue
            if indices:
                input_lst.append(row[indices:])
                indices_lst.append(row[:indices])
            else:
                input_lst.append(row)
    if header and not indices :
        return header_row, np.array(input_lst, dtype="float64")
    elif header and indices:
        return header_row[indices:], indices_lst, np.array(input_lst, dtype="float64")
    return np.array(input_lst, dtype="float64")


##########
### MP ###
##########

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

