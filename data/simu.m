clear all
clc

pt='C:\Users\zsy\Desktop\Ê£Óà6¸ö³¡\671\'; 

xx=[493,511,529,556,571,601,604,607,607,610,619,613];
yy=[364,358,358,355,358,349,364,355,346,358,352,352];
g1=zeros(288,288);
for j=1:1:12
    image=imread(strcat(pt, 'BImg', num2str(8052+j), '.jpg' )); 
    image=double(image);
    for m=1:1:96*3
      for n=1:1:96*3  
          g1(m,n)=image(yy(j)-1+m,xx(j)-1+n);
      end
    end
    for ms=1:1:96
        for ns=1:1:96
          sum=g1(ms*3-2,ns*3-2)+g1(ms*3-2,ns*3-1)+g1(ms*3-2,ns*3)+g1(ms*3-1,ns*3-2)+g1(ms*3-1,ns*3-1)+g1(ms*3-1,ns*3)+g1(ms*3,ns*3-2)+g1(ms*3,ns*3-1)+g1(ms*3,ns*3);
          g2(ms,ns)=sum;
          sum=0;
        end
    end
%     figure
%     imshow(g1,[]);
    figure
    imshow(g2,[]);
g2g=mat2gray(g2);
    imwrite(mat2gray(g2), ['b',num2str(8052+j),'.jpg']);
end

% pro=imread('D:\CNN DATA\model+data\ccd1.bmp');
% probig=zeros(96,48);
% for kk=1:1:96
%     for hh=1:1:48
%        probig(kk,hh)=pro(kk,hh);
%     end
% end
% figure
% imshow(uint8(probig),[]);

