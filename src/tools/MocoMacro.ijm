

File.openSequence("C:/Users/gt8mar/Desktop/data/220513/pointer2/");
open("C:/Users/gt8mar/Desktop/data/220513/pointer2/Basler_acA1300-200um__23253950__20220513_155354922_0659.tiff");
selectWindow("Basler_acA1300-200um__23253950__20220513_155354922_0659.tiff");
run("moco ", "value=25 downsample_value=1 template=Basler_acA1300-200um__23253950__20220513_155354922_0659-1.tiff stack=pointer2 log=None plot=[No plot]");
// todo: mkdir 
run("Image Sequence... ", "select=C:/Users/gt8mar/Desktop/data/220513/moco/ dir=C:/Users/gt8mar/Desktop/data/220513/moco/ format=TIFF use");
