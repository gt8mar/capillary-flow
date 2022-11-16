"""
Filename: ij_test.py
--------------------------
This file is for testing pyimagej.
"""
import os
import imagej
import scyjava

SET = 'set_01'
sample = 'sample_001'
input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'A_cropped'),
output_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'B_stabilized'),
template_path = os.path.join(input_folder, 'Basler_acA1300-200um__23253950__20220513_155354922_0659.tiff')

plugins_dir = 'C:\\Users\\gt8mar\\Documents\\Marcus\\Fiji.app\\plugins'
scyjava.config.add_option(f'-Dplugins.dir={plugins_dir}')

ij= imagej.init('2.5.0')  # 'C:\\Users\\gt8mar\\Documents\\Marcus\\Fiji.app'
print(ij.getVersion())

macro = """
#@ String name
#@ int age
#@ String city
#@output Object greeting
greeting = "Hello " + name + ". You are " + age + " years old, and live in " + city + "."
"""
macroMoco = """
#@ String input_folder
#@ String template_path
#@ File.openSequence();
#@ open("C:/Users/gt8mar/Desktop/data/220513/pointer2/Basler_acA1300-200um__23253950__20220513_155354922_0659.tiff");

#@output Object greeting



"""
args = {
    'name': 'Chuckles',
    'age': 13,
    'city': 'Nowhere'
}
argsMoco = {
    'input_folder': input_folder,
    'output_folder': output_folder,
    'template_path': 13,
}

result = ij.py.run_macro(macro, args)
print(result.getOutput('greeting'))