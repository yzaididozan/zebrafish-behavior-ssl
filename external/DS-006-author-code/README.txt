
Title: Zebrafish larvae exploration and aversive chemotaxis dataset

PI Contact on experimental data: Dr. Claire Wyart, ICM, Paris, France, claire.wyart@icm-institute.org
PI Contact on computational analysis: Dr. Gautam Reddy, Harvard University, Cambridge, MA, USA, gautam_nallamala@fas.harvard.edu

Dates of data collection: June 2018 to December 2019. 

The dataset contains MATLAB files obtained by processing the raw videos using the software ZebraZoom (see zebrazoom.org). Each MATLAB file corresponds to a single experiment. Each experiment has 12 fish in 12 independent wells. See instructions below for more information about the variables in the MATLAB files and for manually extracting the MATLAB data in Python. 

The folder "Catamaran_pH_2bTxtOnly" contains data from 8 experiments in folders labeled as Catamaran_pH_2b_tx where x = 1 to 8. The subfolders for each experiment has the corresponding MATLAB file. 
The folder "Catamaran_pH_2cTxtOnly" contains data from 8 experiments in folders labeled as Catamaran_pH_2c_tx where x = 1 to 8. The subfolders for each experiment has the corresponding MATLAB file. 
The folder "resultsForSB1/ZZoutput/" contains data from 10 experiments in folders labeled as Catamaran_pH_1a_tx where x = 1a,1b,1c,2a,2b,2c,3a,3c,4a,4c. The subfolders for each experiment has the corresponding MATLAB file.
The folder "resultsMay2019/Catamaran_pH_2a/" contains data from 7 experiments in folders labeled as Catamaran_pH_1a_tx where x = 1a,2 to 7. The subfolders for each experiment has the corresponding MATLAB file. 

The methodology of data collection is detailed in the paper "A lexical approach for identifying behavioural action sequences". 

The functions to load the matlab files are also included in the software files uploaded to Zenodo, specifically the IPython notebook "Zebrafish_larvae_analysis_acid_data_final.ipynb". 
This notebook has functions to load the dataset and includes the annotation of the swimming environment of the fish in each well of every experiment. 


You can initially load a matlab tracking result file with the following command:

import scipy.io
supstruct = scipy.io.loadmat('resultFile.mat')

Then, you can see the data for the well numWell and the bout numBout using the following command:

supstruct['videoDataResults'][0][0][0][0][numWell][0][0][0][numBout]

For example, if you want to look at the data for the second bout in the third well, you can type:

bouts_temp = supstruct['videoDataResults'][0][0][0][0][2][0][0][0][1]

You can then, for example, plot the tail angle with the following command:

import matplotlib.pyplot as plt
plt.plot(supstruct['videoDataResults'][0][0][0][0][2][0][0][0][1]["TailAngle_smoothed"])
plt.show()

The full list of parameters available for each bout is:

'FishNumber' : Fish number in the well. If there's only one fish per well, this number will be 0.

'BoutStart' : Frame at which the bout started.

'BoutEnd' : Frame at which the bout ended.

'TailAngle_Raw' : Tail angle over time for the bout, without any smoothing.

'HeadX' : Position on the x axis of the center of the head of the animal, for each frame.

'HeadY' : Position on the y axis of the center of the head of the animal, for each frame.

'Heading_raw' : Value of the main angle of the head of the animal, for each frame, without any smoothing.

'Heading' : Value of the main angle of the head of the animal, for each frame, with smoothing.

'TailX_VideoReferential' : Position on the x axis of each of the points along the tail of the animal, for each frame.

'TailY_VideoReferential' : Position on the y axis of each of the points along the tail of the animal, for each frame.

'TailX_HeadingReferential' : Position on the x axis of each of the points along the tail of the animal, for each frame, when changing the referential such that the head of the animal is at the position (0, 0) and the y axis is aligned with the heading.

'TailY_HeadingReferential' : Position on the y axis of each of the points along the tail of the animal, for each frame, when changing the referential such that the head of the animal is at the position (0, 0) and the y axis is aligned with the heading.

'TailAngle_smoothed' : Tail angle over time for the bout, with smoothing.

'Bend_TimingAbsolute' : List of frames at which the tail angle reached a local maximum or minimum.
'Bend_Timing' : List of frames at which the tail angle reached a local maximum or minimum, with frame 0 being set at the beginning of the bout.

'Bend_Amplitude' : List of amplitudes of the tail angles, for each of the local maximum or minimum reached by the tail angle.

'param' : legacy parameter to ignore.

'Head_Index_Interpolation' : legacy parameter to ignore.

'Tail_Index_Interpolation' : legacy parameter to ignore.




