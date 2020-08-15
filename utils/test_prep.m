clc;
close all;
%�Ҷ��ݶȲ���ͼ��
I=im2double(imread('E:\frame003.png'));
    
%===============================�Ҷ��ݶ�===================================
%3*3�ṹԪ��
n=3;
B=ones(3,3);
n_l=floor(n/2);
%�Ա߽�ͼ�������䣬Ŀ����Ϊ�˴���߽��,������ñ߽羵����չ
I_pad=padarray(I,[n_l,n_l],'symmetric');
[M,N]=size(I);
J_Erosion=zeros(M,N);
J_Dilation=zeros(M,N);
J_Gradient=zeros(M,N);
for i=1:M
    for j=1:N
        %���ͼ���ӿ�����
        Block=I_pad(i:i+2*n_l,j:j+2*n_l);
        C=Block.*B;
        C=C(:);
        %��ʴ����
        J_Erosion(i,j)=min(C);
        %���Ͳ���
        J_Dilation(i,j)=max(C);
        %�Ҷ��ݶ�
        J_Gradient(i,j)=J_Dilation(i,j)-J_Erosion(i,j);
    end
end
imshow(J_Gradient)