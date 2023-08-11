// Specify path to directory
parentDirectory = getDirectory("Choose a parent directory");

// Get a list of all files and folders in the parent directory
allFiles = getFileList(parentDirectory);

// Loop through each file/folder in the list
for (i=0; i<allFiles.length; i++){ //allFiles.length
	print(allFiles[i]);
	vidFolder = parentDirectory + allFiles[i];
	 if (File.isDirectory(vidFolder)) {
	 	File.openSequence(vidFolder);
        // Get a list of all images in the current directory:
        imageFiles = getFileList(vidFolder);
        open(vidFolder + imageFiles[0]);
        folderName = allFiles[i].substring(0, lengthOf(allFiles[i])-1);
        File.makeDirectory(vidFolder + "moco");
        File.makeDirectory(vidFolder + "metadata");
        mocoInput = "value=50 downsample_value=2 template="+imageFiles[0]+" stack="+folderName+" log=[Generate log file] plot=[No plot]";
        run("moco ", mocoInput);
        metadataPath = vidFolder+"metadata/Results.csv";
        sequencePath = "dir="+vidFolder+"moco/ format=TIFF name="+folderName+"_moco_";
		saveAs("Results", metadataPath);
		run("Image Sequence... ", sequencePath);
		
		selectWindow(folderName);
		close();
		selectWindow(imageFiles[0]);
		close();
		selectWindow("New Stack");
		close();
		run("Clear Results");



//inputFile = parentDirectory + allFiles[i]+imageFiles[i];
//        print(inputFile);

//        File.openSequence("C:/Users/gt8mar/Desktop/data/230414/vid1/");
        // Loop through each image file in the list
//    	for (j = 0; j < imageFiles.length; j++) {
//    		if (j == 0) {
//            // If this is the first image in the stack, create a new stack
//            stack = File.makeStack(parentDirectory + allFiles[i] + "/" + imageFiles[j]);
//            }; 
//            else {
//          	// If this is not the first image in the stack, add it to the existing stack
//            stack = File.openSequence(parentDirectory + allFiles[i] + "/" + imageFiles[j]);
//            stack = stack.makeStack(stack.width, stack.height, stack.getStackSize());
//            };
////        	print(imageFiles[j]);
//			print(stack);
//    	};
    };
};
    // Check if the current file/folder is a directory
   
    
//        // Check if the current file is an image
//        if (endsWith(imageFiles[j], ".tiff")) {
//            // Open the image and add it to the stack
            
//        }
//    
//    // Do something with the image stack (e.g. display it)
//        stack.show();
//    }
////    selectWindow("vid1");
////    File.openSequence("C:/Users/gt8mar/Desktop/data/230414/vid1/");
////    run("Duplicate...", "use");
////    run("moco ", "value=50 downsample_value=2 template=Basler_acA1440-220um__40131722__20230414_092243769_0000.tiff stack=vid1 log=[Generate log file] plot=[Plot RMS]");
////    saveAs("Results", "C:/Users/gt8mar/Desktop/data/230414/vid1/metadata/Results.csv");
////    close();
////    run("Image Sequence... ", "select=C:/Users/gt8mar/Desktop/data/230414/vid1/moco/ dir=C:/Users/gt8mar/Desktop/data/230414/vid1/moco/ format=TIFF name=vid1_moco_");
////    close();
////    close();
////    selectWindow("vid1");
////    close();
////    open(parentDirectory + allFiles[i] + "/myImage.tif");
//}

