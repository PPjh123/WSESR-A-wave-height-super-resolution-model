clear
clc
load('D:\DATA\Wave_height_super_resolution\Data\ERA5-3h\S3_2021_2024.mat')
for i=1:size(swh,3)
    swh_wt(:,:,:,i)=WT2D(swh(:,:,i));
end
save ERA5-3h-wt\S3_2021_2024.mat swh_wt%%%%%


function result=WT2D(input)
wavelet = 'db4';
level = 2;
% 分解
[C, S] = wavedec2(input, level, wavelet);
cA2 = appcoef2(C,S,wavelet,2); %提取2级低频成分
cH2 = detcoef2('h',C,S,2); %提取2级高频成分，h，v，d分别代表水平、垂直、对角线
cV2 = detcoef2('v',C,S,2);
cD2 = detcoef2('d',C,S,2); 

cH1 = detcoef2('h',C,S,1);%提取1级高频成分
cV1 = detcoef2('v',C,S,1);
cD1 = detcoef2('d',C,S,1);
% 重构
result(:,:,1)= wrcoef2('a',C,S,wavelet,2); %2级系数重构，a、h、v、d分别代表低频，高频水平、竖直、对角线
result(:,:,2) = wrcoef2('h',C,S,wavelet,2);  
result(:,:,3)= wrcoef2('v',C,S,wavelet,2);  
result(:,:,4)= wrcoef2('d',C,S,wavelet,2); 

result(:,:,5) = wrcoef2('a',C,S,wavelet,1); 
result(:,:,6) = wrcoef2('h',C,S,wavelet,1); 
result(:,:,7) = wrcoef2('v',C,S,wavelet,1);
result(:,:,8) = wrcoef2('d',C,S,wavelet,1); 

end
