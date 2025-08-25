clear
clc
%data loading
ncdisp('Copernicus-3h\S3-2021-2024.nc')
data=ncread('Copernicus-3h\S3-2021-2024.nc','VHM0');%%%%%%%%
% area cut
swh= permute(data, [2, 1, 3]);
swh=flipud(swh);
swh=swh(2:end,2:end,:);
save Copernicus-3h\S3_2021_2024.mat swh%%%%%%%%
% % check
% X=data(:,:,1);
% imagesc(X)
%% Nan value padding
clear
clc
load 'Copernicus-3h\S3_2021_2024.mat'%%%%%%%%
swh(isnan(swh)) = 0;
save Copernicus-3h\S3_2021_2024.mat swh%%%%%%%%
%check
X=swh(:,:,1);
nan_value= any(isnan(X),'all');

imagesc(X)
