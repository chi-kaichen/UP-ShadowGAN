%==========================================================================
% J. Yan, J. Li, X. Fu, "No-Reference Quality Assessment of Contrast-Distorted Images using Contrast Enhancement"
% 
% Please try your own contrast distorted images with different levels.
% Larger predicted score means better contrast quality.
% This model was trained by all the images in CCID2104 database using
% LIBSVM.
%==========================================================================

clear;
clc;

load results

abs(corr(CID2013_sge, CID2013_mos(:,1), 'type','spearman'))
abs(corr(CCID2014_sge,CCID2014_mos(:,1),'type','spearman'))
abs(corr(CSIQ_sge, CSIQ_dmos,   'type','spearman'))
abs(corr(TID2013_sge, TID2013_mos, 'type','spearman'))
