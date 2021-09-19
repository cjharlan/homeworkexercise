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

     V = niftiread('ICBM_Template.nii.gz');
     brain = zeros(181,217,181,4);
     brain(:,:,:,1) = V;
     brain(:,:,:,2) = V;
     brain(:,:,:,3) = V;
     brain(:,:,:,4) = V;
     brainpatch = brain(49:49+64-1,67:67+64-1,49:49+64-1,:);
     size(brainpatch)
     analyzeNetwork(net)
     tempSeg = semanticseg(brainpatch,net);

     figure(1)
     imagesc(uint8(tempSeg(:,:,32)));
     figure(2)
     imagesc(brainpatch(:,:,32));

