  clc
  clear
clear all
close all
tic
%% 20190910
% ECTCNN train data/ input X  12views bunsen phantom

%%ͼƬ��������
% pt='D:\CNN DATA\phantom for U-Net\train data\figures\'; %train sets
% pt='D:\CNN DATA\phantom for U-Net\test data\figures\'; %test sets
  pt='C:\Users\zsy\Desktop\ʣ��6����\671\'; %test sets
img_path_list=dir(strcat(pt, '*.jpg'));

nameCell=cell(length(img_path_list),1);
l=length(img_path_list);
P = cell(1,l);%����һ��ϸ�����飬���ڴ������txt�ļ�
for  i=1:l
    nameCell{i}=img_path_list(i).name;
end

img_num=length(img_path_list);
img_path_list=sort_nat(nameCell);
%����CNN��������X
%һ��˲ʱͬһ���12����������ݣ�Ϊһ��.mat ��СΪ150*10
cominput=zeros(1,96,96,10);

for j=1:1:1%250��˲ʱ
      X0=zeros(96,96,10);
for k=1:1:10    %12������
       image=imread(strcat(pt, 'b', num2str(8052+k), '.jpg' ));
%       figure
%       imshow(image,[ ]);
    for m=1:1:96
        for n=1:1:96
         X0(m,n,k)=image(m,n);   
        end
    end
%       figure
%       imshow(X0(:,:,k),[ ]);
end
cominput(j,:,:,:)=X0;

% eval(['S',num2str(j-1),'=','X0',';']);
%  filename=strcat('S',num2str(j-1),'.mat');
%  save(filename,'X0');

end
save ('C:\Users\zsy\Desktop\ʣ��6����\671\cominputb10.mat','cominput','-v7.3')
disp('Complete');

t1=toc
