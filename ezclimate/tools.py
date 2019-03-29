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
    d = find_path(file_name)
    if index is not None:
        index.extend(lst)
        output_lst = list(zip(*index))
    else:
        output_lst = list(zip(*lst))

    with open(d, open_as) as f:
        writer = csv.writer(f, delimiter=delimiter)
        if start_char is not None:
            writer.writerow([start_char])
        if header:
            writer.writerow(header)
        for row in output_lst:
            #print('***DEBUG -- printing row and type(row):',row,type(row))
            writer.writerow(row)

def write_columns_to_existing(lst, file_name, header="", delimiter=';'):
    d = find_path(file_name)
    #print('***In WCTE, lst= ',lst)
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
        #print('***In WCTE, row+ = ', row)
        all_lst.append(row)
        n = len(lst)
        i = 0
        for row in reader:
            #print('***In WCTE, row ',i,'- = ', row)
            if nested_list:
                row.extend(lst[i])
            else:
                row.append(lst[i])
            #print('***In WCTE, row ',i,'+ = ', row)
            all_lst.append(row)
            i += 1
    #print('***In WCTE, all_lst =',all_lst)
    with open(d, 'w') as foutput:
        writer = csv.writer(foutput, delimiter=delimiter)
        writer.writerows(all_lst)
            
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

