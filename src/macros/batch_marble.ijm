macro "batch_marble_auto" {
    setBatchMode(true);

    rootDir = getArgument();
    regDir  = rootDir + File.separator + "regions" + File.separator;
    maskDir = rootDir + File.separator + "masks"   + File.separator;
    File.makeDirectory(maskDir);

    list = getFileList(regDir);
    for (idx = 0; idx < list.length; idx++) {
        if (!endsWith(list[idx], "_crop.png")) continue;

        open(regDir + list[idx]);
        open(rootDir + File.separator + "regions" + File.separator +
            replace(list[idx], "_crop.png", "_roi.png"));    
        run("Create Selection");                  
        selectWindow(list[idx]);                  
        run("Restore Selection");
        setBackgroundColor(0,0,0); run("Clear Outside");     
        close("*_roi.png");                       

        run("Subtract Background...", "rolling=25");
        run("8-bit");

        AUTO_THRESHOLD = 5000;
        getRawStatistics(pixcount);
        limit = pixcount/10;
        threshold = pixcount/AUTO_THRESHOLD;
        nBins = 256;
        getHistogram(values, histA, nBins);
        h = -1; found = false;
        do{
            counts = histA[++h];
            if (counts > limit) counts = 0;
            found = counts > threshold;
        }while((!found)&&(h < histA.length-1));
        hmin = values[h];

        h = histA.length;
        do{
            counts = histA[--h];
            if (counts > limit) counts = 0;
            found = counts > threshold;
        }while((!found)&&(h > 0));
        hmax = values[h];
        setMinAndMax(hmin, hmax);
        run("Apply LUT");

        run("Auto Local Threshold", "method=Otsu radius=5 white");
        run("Enlarge...", "enlarge=-15");
        setBackgroundColor(0,0,0); run("Clear Outside");

        run("Analyze Particles...", "size=14-Infinity show=[Masks] include");
        saveAs("PNG", maskDir + list[idx]);   
        close("*");
    }
}