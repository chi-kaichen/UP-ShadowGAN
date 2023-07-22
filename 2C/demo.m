clear;
clc;
addpath('utils', 'data', 'images');
for i = 1:50
img = imread([num2str(i,'%d'),'.png']);
cform = makecform('srgb2lab');
Img_lab = applycform(img, cform);
Img_a = double(Img_lab(:,:,2))./255;
Img_b = double(Img_lab(:,:,3))./255;
Img_Chr = sqrt(Img_a(:).^2+Img_b(:).^2); 
Aver_Chr = mean(Img_Chr);
sigma = sqrt(mean((abs(1-(Aver_Chr./Img_Chr).^2))));
omega = CEIQ(img);
CC = sigma - 0.03 * omega;
str = num2str(CC);
disp(str);
end