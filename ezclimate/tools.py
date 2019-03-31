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
    """
    Append data column or columns to an existing CSV file.
    Parameters
    ----------
    lst : list or np.ndarray
        contains column or columns to append; if single column, a flat list;
        if multiple columns, a nested list or list of 1D np.ndarrays
    file_name : str
        file name to append to
    header : str or list
        header for the added columns; if adding just one column, a
        string, otherwise a List[str]
    delimiter : str
        delimiter for the CSV file (default semi-colon)
    """

    # Is there one added column, or multiple? If first item list or np.ndarrary
    # (i.e. nested_list) => multiple; otherwise => single added column. If
    # nested / multiple columns, transpose into rows for row-oriented output.
    is_nested_list = lst and (isinstance(lst[0], list) or
                                isinstance(lst[0], np.ndarray))
    if is_nested_list:
        lst = list(zip(*lst))   # transpose columns -> rows

    file_path = find_path(file_name)
    output_rows = []

    # read and extend input
    with open(file_path, 'r') as finput:
        reader = csv.reader(finput, delimiter=delimiter)

        # extend header row
        row = next(reader)
        row.extend(header if is_nested_list else [header])
        output_rows.append(row)

        # extend rest of the rows
        for i, row in enumerate(reader):
            row.extend(lst[i] if is_nested_list else [lst[i]])
            output_rows.append(row)

    # emit output, overwriting original file
    with open(file_path, 'w') as foutput:
        writer = csv.writer(foutput, delimiter=delimiter)
        writer.writerows(output_rows)
            
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

