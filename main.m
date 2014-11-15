%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run this script to obtain the classification results
% Author: Bharat Tak
% Date : 15 nov 2014

%% Read and parse data 
clc
clear all
close all


% path to folder containing the dataset
dataset_path = '/project2/';

training_data = csvread(strcat(dataset_path,'training.csv'));
validation_data = csvread(strcat(dataset_path, 'validation.csv'));
testing_data = csvread(strcat(dataset_path, 'testing.csv'));

training_vec = training_data(:,1:27);
training_class = training_data(:,end);

validation_vec = validation_data;
validation_class = zeros(size(validation_vec,1),1);

testing_vec = testing_data;
testing_class = zeros(size(testing_vec,1),1);

%% SVM from matlab
%  Ref: http://www.mathworks.com/help/stats/svmtrain.html
%       http://www.mathworks.com/help/stats/svmclassify.html

svm_struct = svmtrain(training_vec,training_class,'kernel_function','rbf',...
                    'kktviolationlevel',0.0,'rbf_sigma',0.5);
validation_class = svmclassify(svm_struct,validation_vec);
testing_class = svmclassify(svm_struct,testing_vec);

%% Write output file
csvwrite(strcat('validation_class'),validation_class);
csvwrite(strcat('testing_class'),testing_class);