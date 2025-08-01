macro "batch_colour_auto" {
    setBatchMode(true);
    SHRINK_PX = 10;

    rootDir = getArgument();                                  
    regDir  = rootDir + File.separator + "regions" + File.separator;
    leanDir = rootDir + File.separator + ".." + File.separator + "marbling" + File.separator + "masks" + File.separator;
    resDir  = rootDir + File.separator + "results" + File.separator;
    File.makeDirectory(resDir);

    C_R = newArray(7); C_G = newArray(7); C_B = newArray(7);
    mid = newArray(6);
    vHist = newArray(256); cHist = newArray(256); histA = newArray(256);

    list = getFileList(regDir);
    for (i = 0; i < list.length; i++) {
        if (!endsWith(list[i], "_colour.png")) continue;
        base = replace(list[i], "_colour.png", "");

        stdPath  = regDir + base + "_std.txt";
        if (!File.exists(stdPath)) {
            print("Skipping " + base + ": missing standards file -> " + stdPath);
            close("*");
            continue;
        }

        stdTxt   = File.openAsString(stdPath);
        stdLines = split(stdTxt, "\r?\n");

        while (stdLines.length > 0 && trim(stdLines[stdLines.length - 1]) == "") {
            stdLines = Array.slice(stdLines, 0, stdLines.length - 1);
        }

        if (stdLines.length < 7) {
            print("Skipping " + base + ": standards file has fewer than 7 lines -> " + stdPath);
            close("*");
            continue;
        }

        for (s = 0; s < 7; s++) {
            vals = split(trim(stdLines[s]), ",");
            if (vals.length < 3) {
                print("Skipping " + base + ": bad standard line " + s + " -> " + stdLines[s]);
                close("*");
                continue;
            }
            C_R[s] = parseInt(vals[0]);
            C_G[s] = parseInt(vals[1]);
            C_B[s] = parseInt(vals[2]);
        }

        if (C_G[0] < C_G[6]) {
            R2=newArray(7); G2=newArray(7); B2=newArray(7);
            for (s=0; s<7; s++) { idx=6-s; R2[s]=C_R[idx]; G2[s]=C_G[idx]; B2[s]=C_B[idx]; }
            for (s=0; s<7; s++) { C_R[s]=R2[s]; C_G[s]=G2[s]; C_B[s]=B2[s]; }
        }

        open(regDir + base + "_roi.png"); rename("roi");
        run("8-bit"); setOption("BlackBackground", true); run("Convert to Mask");
        for (t = 0; t < SHRINK_PX; t++) run("Erode");

        open(leanDir + base + "_crop.png"); rename("marb");
        run("8-bit"); setOption("BlackBackground", true); run("Convert to Mask");
        run("Invert");

        run("Image Calculator...", "image1=roi operation=AND image2=marb create");
        rename("lean_mask");
        close("roi"); close("marb");

        open(regDir + base + "_G.png"); rename("Gchan"); run("8-bit");
        if (!isOpen("Gchan")) {
            print("Skipping " + base + ": could not open G plane");
            close("*");
            continue;
        }

        selectWindow("Gchan"); run("Duplicate...", "title=Gviz");
        selectWindow("Gviz");
        run("Subtract Background...", "rolling=25");
        AUTO_THRESHOLD = 5000; getRawStatistics(pixcount);
        limit = pixcount/10; threshold = pixcount/AUTO_THRESHOLD;
        getHistogram(vHist, histA, 256);
        h=-1; do{cnt=histA[++h]; if(cnt>limit) cnt=0;}while(cnt<=threshold && h<histA.length-1);
        hmin=vHist[h];
        h=histA.length; do{cnt=histA[--h]; if(cnt>limit) cnt=0;}while(cnt<=threshold && h>0);
        hmax=vHist[h]; setMinAndMax(hmin, hmax); run("Apply LUT");

        for (s=0; s<6; s++) mid[s] = floor((C_G[s] + C_G[s+1]) / 2);
        b0=mid[5]; b1=mid[4]; b2=mid[3]; b3=mid[2]; b4=mid[1]; b5=mid[0];

        selectWindow("Gchan"); run("Duplicate...", "title=LABEL");
        changeValues(0,      b0, 7);
        changeValues(b0+1,   b1, 6);
        changeValues(b1+1,   b2, 5);
        changeValues(b2+1,   b3, 4);
        changeValues(b3+1,   b4, 3);
        changeValues(b4+1,   b5, 2);
        changeValues(b5+1,   255, 1);
        run("Image Calculator...", "image1=LABEL operation=AND image2=lean_mask create");
        rename("LABEL_LEAN"); close("LABEL");

        selectWindow("LABEL_LEAN"); getHistogram(vHist, cHist, 256);
        leanPixels = 0; for (k=1; k<=7; k++) leanPixels += cHist[k];

        run("Duplicate...", "title=C_R");
        selectWindow("LABEL_LEAN"); run("Duplicate...", "title=C_G");
        selectWindow("LABEL_LEAN"); run("Duplicate...", "title=C_B");
        selectWindow("C_R"); for (s=0; s<7; s++) changeValues(s+1, s+1, C_R[s]);
        selectWindow("C_G"); for (s=0; s<7; s++) changeValues(s+1, s+1, C_G[s]);
        selectWindow("C_B"); for (s=0; s<7; s++) changeValues(s+1, s+1, C_B[s]);
        selectWindow("C_G"); w=getWidth(); h=getHeight(); run("8-bit");
        selectWindow("C_R"); run("8-bit"); if (getWidth()!=w || getHeight()!=h) run("Canvas Size...", "width="+w+" height="+h+" position=Top-Left");
        selectWindow("C_B"); run("8-bit"); if (getWidth()!=w || getHeight()!=h) run("Canvas Size...", "width="+w+" height="+h+" position=Top-Left");
        run("Merge Channels...", "red=C_R green=C_G blue=C_B create");
        rename(base + "_Canadian_LUT"); saveAs("PNG", resDir + base + "_Canadian_LUT.png");

        run("Clear Results");
        for (s=0; s<7; s++) {
            setResult("image_id",   s, base);
            setResult("Standard",   s, "CdnStd"+s);
            setResult("CdnCount",   s, cHist[s+1]);
            pct = 0.0; if (leanPixels > 0) pct = (cHist[s+1]*100.0)/leanPixels;
            setResult("CdnPercent", s, pct);
        }
        updateResults();
        saveAs("Results", resDir + base + "_colour.xls");

        close("*");
    }
}