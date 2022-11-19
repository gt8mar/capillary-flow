
dir = getDirectory("Choose a Directory");
//file_list = getFileList(dir);

template = getDirectory("Choose a Directory");

setBatchMode(true);

File.openSequence(dir);

open(template);
//selectWindow("pointer2");
//selectWindow("Basler_acA1300-200um__23253950__20220513_155354922_0659.tiff");
run("moco ", "value=25 downsample_value=2 template=template stack=dir log=None plot=[No plot]");
run("Image Sequence... ", "select=C:/Users/gt8mar/Desktop/data/220513/vid/ dir=C:/Users/gt8mar/Desktop/data/220513/vid/ format=TIFF");
