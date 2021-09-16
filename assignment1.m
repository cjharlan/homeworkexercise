% created from https://www.mathworks.com/help/images/segment-3d-brain-tumor-using-deep-learning.html
% >> ver -support
% -----------------------------------------------------------------------------------------------------
% MATLAB Version: 9.8.0.1417392 (R2020a) Update 4
% MATLAB License Number: 68666
% Operating System: Linux 4.4.0-127-generic #153-Ubuntu SMP Sat May 19 10:58:46 UTC 2018 x86_64
% Java Version: Java 1.8.0_202-b08 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode
% -----------------------------------------------------------------------------------------------------
% MATLAB                                                Version 9.8         (R2020a)      License 68666
% Simulink                                              Version 10.1        (R2020a)      License 68666
% Bioinformatics Toolbox                                Version 4.14        (R2020a)      License 68666
% Computer Vision Toolbox                               Version 9.2         (R2020a)      License 68666
% Curve Fitting Toolbox                                 Version 3.5.11      (R2020a)      License 68666
% Deep Learning Toolbox                                 Version 14.0        (R2020a)      License 68666
% Image Acquisition Toolbox                             Version 6.2         (R2020a)      License 68666
% Image Processing Toolbox                              Version 11.1        (R2020a)      License 68666
% MATLAB Compiler                                       Version 8.0         (R2020a)      License 68666
% MATLAB Compiler SDK                                   Version 6.8         (R2020a)      License 68666
% Optimization Toolbox                                  Version 8.5         (R2020a)      License 68666
% Parallel Computing Toolbox                            Version 7.2         (R2020a)      License 68666
% Signal Processing Toolbox                             Version 8.4         (R2020a)      License 68666
% Statistics and Machine Learning Toolbox               Version 11.7        (R2020a)      License 68666
% Symbolic Math Toolbox                                 Version 8.5         (R2020a)      License 68666
% Wavelet Toolbox                                       Version 5.4         (R2020a)      License 68666
% 
% references:
%    https://www.mathworks.com/matlabcentral/answers/427468-how-does-semanticseg-command-work-on-images-larger-than-what-the-network-was-trained-with
%    https://www.mathworks.com/help/deeplearning/ref/activations.html
clear all 
close all

% load nifti functions
addpath nifti

%% Download Pretrained Network and Sample Test Set
% Optionally, download a pretrained version of 3-D U-Net and five sample test 
% volumes and their corresponding labels from the BraTS data set [3]. The pretrained 
% model and sample data enable you to perform segmentation on test data without 
% downloading the full data set or waiting for the network to train.

trained3DUnet_url = 'https://www.mathworks.com/supportfiles/vision/data/brainTumor3DUNet.mat';
sampleData_url = 'https://www.mathworks.com/supportfiles/vision/data/sampleBraTSTestSet.tar.gz';

imageDir = fullfile(tempdir,'BraTS');
if ~exist(imageDir,'dir')
    mkdir(imageDir);
end
downloadTrained3DUnetSampleData(trained3DUnet_url,sampleData_url,imageDir);

% return a pretrained 3-D U-Net network.
inputPatchSize = [132 132 132 4];
outPatchSize = [44 44 44 2];
load(fullfile(imageDir,'trained3DUNet','brainTumor3DUNet.mat'));
%analyzeNetwork(net)

% You can now use the U-Net to semantically segment brain tumors.
%% Perform Segmentation of Test Data
% load five volumes for testing.
volLocTest = fullfile(imageDir,'sampleBraTSTestSet','imagesTest');
%volLocTest = fullfile(niftiread('ICBM_Template.nii.gz'));
lblLocTest = fullfile(imageDir,'sampleBraTSTestSet','labelsTest');
classNames = ["background","tumor"];
pixelLabelID = [0 1];

%% 
% Crop the central portion of the images and labels to size 128-by-128-by-128 
% voxels by using the helper function |centerCropMatReader|. This function is 
% attached to the example as a supporting file. The |voldsTest| variable stores 
% the ground truth test images. The |pxdsTest| variable stores the ground truth 
% labels.

windowSize = [128 128 128];
volReader = @(x) centerCropMatReader(x,windowSize);
labelReader = @(x) centerCropMatReader(x,windowSize);
voldsTest = imageDatastore(volLocTest, ...
    'FileExtensions','.mat','ReadFcn',volReader);
pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',labelReader);
%% 
% For each test image, add the ground truth image volumes and labels to cell 
% arrays. Use the trained network with the <docid:vision_ref#mw_bbecb1af-a6c9-43d1-91f5-48607edc15d1 
% |semanticseg|> function to predict the labels for each test volume.
% 
% After performing the segmentation, postprocess the predicted labels by labeling 
% nonbrain voxels as |1|, corresponding to the background. Use the test images 
% to determine which voxels do not belong to the brain. You can also clean up 
% the predicted labels by removing islands and filling holes using the <docid:images_ref#bvb_85o-1 
% |medfilt3|> function. |medfilt3| does not support categorical data, so cast 
% the pixel label IDs to |uint8| before the calculation. Then, cast the filtered 
% labels back to the categorical data type, specifying the original pixel label 
% IDs and class names.

id=1;
     V = niftiread('ICBM_Template.nii.gz');
     brain = zeros(181,217,181,4);
     brain(:,:,:,1) = V;
     brain(:,:,:,2) = V;
     brain(:,:,:,3) = V;
     brain(:,:,:,4) = V;
    % brain = brain(:,18:198,:,:);
     brain = brain(59:122,77:140,59:122,:);
while hasdata(voldsTest)
    disp(['Processing test volume ' num2str(id)])
    
    tempGroundTruth = read(pxdsTest);
    groundTruthLabels{id} =  tempGroundTruth{1};
    
   %  vol{id} = read(voldsTest);

 %    vol{id} = niftiread('ICBM_Template.nii.gz');
     vol{id} = brain;
     tempSeg = semanticseg(brain,net);

    % Get the non-brain region mask from the test image.
    volMask = vol{id}(:,:,:,1)==0;
    % Set the non-brain region of the predicted label as background.
    tempSeg(volMask) = classNames(1);
    % Perform median filtering on the predicted label.
    tempSeg = medfilt3(uint8(tempSeg)-1);
    % Cast the filtered label to categorial.
    tempSeg = categorical(tempSeg,pixelLabelID,classNames);
    predictedLabels{id} = tempSeg;
    id=id+1;
end
%% Compare Ground Truth Against Network Prediction
% Select one of the test images to evaluate the accuracy of the semantic segmentation. 
% Extract the first modality from the 4-D volumetric data and store this 3-D volume 
% in the variable |vol3d|.

volId = 2;
vol3d = vol{volId}(:,:,:,1);
%% 
% Display in a montage the center slice of the ground truth and predicted labels 
% along the depth direction.

zID = size(vol3d,3)/2;
zSliceGT = labeloverlay(vol3d(:,:,zID),groundTruthLabels{volId}(:,:,zID));
zSlicePred = labeloverlay(vol3d(:,:,zID),predictedLabels{volId}(:,:,zID));

figure
title('Labeled Ground Truth (Left) vs. Network Prediction (Right)')
montage({zSliceGT;zSlicePred},'Size',[1 2],'BorderSize',5) 
%% 
% Display the ground-truth labeled volume using the <docid:images_ref#mw_7a40592d-db0e-4bdb-9ba3-446bd1715151 
% |labelvolshow|> function. Make the background fully transparent by setting the 
% visibility of the background label (|1|) to |0|. Because the tumor is inside 
% the brain tissue, make some of the brain voxels transparent, so that the tumor 
% is visible. To make some brain voxels transparent, specify the volume threshold 
% as a number in the range [0, 1]. All normalized volume intensities below this 
% threshold value are fully transparent. This example sets the volume threshold 
% as less than 1 so that some brain pixels remain visible, to give context to 
% the spatial location of the tumor inside the brain.

figure
h1 = labelvolshow(groundTruthLabels{volId},vol3d);
h1.LabelVisibility(1) = 0;
h1.VolumeThreshold = 0.68;
%% 
% For the same volume, display the predicted labels.

figure
h2 = labelvolshow(predictedLabels{volId},vol3d);
h2.LabelVisibility(1) = 0;
h2.VolumeThreshold = 0.68;
%% 
% This image shows the result of displaying slices sequentially across the entire 
% volume.
% 
% %% Quantify Segmentation Accuracy
% Measure the segmentation accuracy using the <docid:images_ref#mw_1ee709d7-bf6b-4ac9-8f5d-e7caf72497d4 
% |dice|> function. This function computes the Dice similarity coefficient between 
% the predicted and ground truth segmentations.

diceResult = zeros(length(voldsTest.Files),2);

for j = 1:length(vol)
    diceResult(j,:) = dice(groundTruthLabels{j},predictedLabels{j});
end
%% 
% Calculate the average Dice score across the set of test volumes.

meanDiceBackground = mean(diceResult(:,1));
disp(['Average Dice score of background across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceBackground)])
meanDiceTumor = mean(diceResult(:,2));
disp(['Average Dice score of tumor across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceTumor)])
%% 
% The figure shows a <docid:stats_ug#bu180jd |boxplot|> that visualizes statistics 
% about the Dice scores across the set of five sample test volumes. The red lines 
% in the plot show the median Dice value for the classes. The upper and lower 
% bounds of the blue box indicate the 25th and 75th percentiles, respectively. 
% Black whiskers extend to the most extreme data points not considered outliers.
% 
% % 
% If you have Statistics and Machine Learning Toolbox™, then you can use the 
% |boxplot| function to visualize statistics about the Dice scores across all 
% your test volumes. To create a |boxplot|, set the |createBoxplot| parameter 
% in the following code to |true|.

createBoxplot = true;
if createBoxplot
    figure
    boxplot(diceResult)
    title('Test Set Dice Accuracy')
    xticklabels(classNames)
    ylabel('Dice Coefficient')
end
%% Summary
% This example shows how to create and train a 3-D U-Net network to perform 
% 3-D brain tumor segmentation using the BraTS data set. The steps to train the 
% network include:
