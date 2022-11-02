"""
Filename: sort_nicely.py
-------------------------------------------------------------
This file correctly orders misordered files.
by: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import re

def tryint(string):
    """ Check if strings have integers inside """
    try:
        return int(string)
    except:
        return string


def alphanum_key(string):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]"""
    return [tryint(char) for char in re.split('([0-9]+)', string)]


def main(list):
    """ Sort the given list in the way that humans expect """
    list.sort(key=alphanum_key)
    return 0


# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    main()
