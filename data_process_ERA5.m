clear
clc
ncdisp('ERA5-3h\S3-2021.nc')%%%%%%%%%
%data loading and connecting
years=[2021:2024];
data_all=[];
for i=1:length(years)
    data=ncread(['ERA5-3h\S3-',num2str(years(i)),'.nc'],'swh');%%%%%%
    data_single= permute(data, [2, 1, 3]);
    data_all=cat(3,data_all,data_single);
end
%area cut
swh=data_all(2:end,2:end,:);
save ERA5-3h\S3_2021_2024.mat swh%%%%%%%%%%
%% Nan value padding
clear
clc
load 'ERA5-3h\S3_2021_2024.mat'%%%%%%%%
swh(isnan(swh)) = 0;
save ERA5-3h\S3_2021_2024.mat swh%%%%%%%%%%%
%check
X=swh(:,:,1);
nan_value= any(isnan(X),'all');

imagesc(X)
