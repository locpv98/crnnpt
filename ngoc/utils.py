import numpy as np
import os


def get_label_from_file(filename):
    """
        Create all strings by reading lines in specified files
    """
    strings = []

    with open(filename, "r", encoding="utf8") as f:
        lines = [l[0:200] for l in f.read().splitlines() if len(l) > 0]
        count = len(lines)
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        while len(strings) < count:
            if len(lines) >= count - len(strings):
                strings.extend(lines[0: count - len(strings)])
            else:
                strings.extend(lines)
    return strings

def get_train_path_from_file(filename):
    """
        Create all strings by reading lines in specified files
    """
    import os
    p = os.path.dirname(filename)
    strings = []
    
    with open(filename, "r", encoding="utf8") as f:
        lines = [str(p)+'/'+l[0:200] for l in f.read().splitlines() if len(l) > 0]
        count = len(lines)
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        while len(strings) < count:
            if len(lines) >= count - len(strings):
                strings.extend(lines[0: count - len(strings)])
            else:
                strings.extend(lines)
    return strings
