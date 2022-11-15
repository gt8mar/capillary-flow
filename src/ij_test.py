"""
Filename: ij_test.py
--------------------------
This file is for testing pyimagej.
"""

import imagej

ij= imagej.init('2.5.0')

print(ij.getVersion())