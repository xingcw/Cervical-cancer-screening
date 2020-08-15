clc;
close all;
%灰度梯度测试图像
I=im2double(imread('E:\frame003.png'));
    
%===============================灰度梯度===================================
%3*3结构元素
n=3;
B=ones(3,3);
n_l=floor(n/2);
%对边界图进行扩充，目的是为了处理边界点,这里采用边界镜像扩展
I_pad=padarray(I,[n_l,n_l],'symmetric');
[M,N]=size(I);
J_Erosion=zeros(M,N);
J_Dilation=zeros(M,N);
J_Gradient=zeros(M,N);
for i=1:M
    for j=1:N
        %获得图像子块区域
        Block=I_pad(i:i+2*n_l,j:j+2*n_l);
        C=Block.*B;
        C=C(:);
        %腐蚀操作
        J_Erosion(i,j)=min(C);
        %膨胀操作
        J_Dilation(i,j)=max(C);
        %灰度梯度
        J_Gradient(i,j)=J_Dilation(i,j)-J_Erosion(i,j);
    end
end
imshow(J_Gradient)