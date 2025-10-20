% ===============================================================
% Autocorrelation Matrix Generation with Data Normalization [-1, 1]
% Author: [Your Name]
% License: MIT
% Description:
%   1. Normalize all feature columns to range [-1, 1]
%   2. Compute row-wise autocorrelation matrices
%   3. Export each as a grayscale image
% ===============================================================

clear all;
close all;
clc;

% --- Select and read Excel file ---
[Filename, Pathname] = uigetfile('dat.xlsx');
DataPath = [Pathname, Filename];
data = xlsread(DataPath);

% --- Example data for testing (optional) ---
% data = rand(200, 64);  % Example: 200 samples × 64 features

% ===============================================================
% Step 1: Normalize the entire dataset to range [-1, 1]
% ===============================================================
minVals = min(data, [], 1);   % Minimum for each column
maxVals = max(data, [], 1);   % Maximum for each column

% Avoid division by zero
rangeVals = maxVals - minVals;
rangeVals(rangeVals == 0) = 1;

% Apply normalization to [-1, 1]
data_norm = 2 * ((data - minVals) ./ rangeVals) - 1;

% ===============================================================
% Step 2: Compute autocorrelation matrices for each normalized row
% ===============================================================
numRows = size(data_norm, 1);                % Number of samples
normalizedAutocorrMatrices = cell(numRows, 1);  % Cell array for results

for i = 1:numRows
    row = data_norm(i, :);           % Normalized feature row
    autocorrMatrix = row' * row;     % Outer product (NxN matrix)
    normalizedAutocorrMatrices{i} = autocorrMatrix;
end

% ===============================================================
% Step 3: Export each autocorrelation matrix as grayscale image
% ===============================================================
outputDir = 'NS/';         % Output folder
filePrefix = 'image_';     % File prefix
fileFormat = 'bmp';        % Image format
fileSuffix = strcat('.', fileFormat);

% Recreate output directory
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
else
    rmdir(outputDir, 's');
    mkdir(outputDir);
end

for i = 1:numRows
    imgMatrix = cell2mat(normalizedAutocorrMatrices(i));
    imwrite(mat2gray(imgMatrix), ...
            strcat(outputDir, filePrefix, num2str(i), fileSuffix), ...
            fileFormat);
end

fprintf("✅ All normalized autocorrelation matrices have been saved to '%s'\n", outputDir);
