function  evol_snake(points,seg)
%EVOL_SNAKE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

I =  imread('C:\Users\chan\Desktop\LIC\data\CBCT_PNG\4\283.png');
% ��Ϊ��ɫ��ת��Ϊ�Ҷ�
if(size(I,3)==3), I=rgb2gray(I); end
% ת��Ϊ˫������
Igs = im2double(I);
%figure(1),imshow(I);
%---------------------------
%        ��˹�˲�
%---------------------------
sigma=1.0;
% �����ض���ʽ�Ķ�ά��˹�˲���H
H = fspecial('gaussian',ceil(3*sigma), sigma);
% ��ͼ����и�˹�˲�,���غ�I�ȴ�С����
Igs = filter2(H,Igs,'same');


% =========================================================================
%                     Snakes�㷨ʵ�ֲ���
% =========================================================================
NIter =1000; % ��������
alpha=0.2; %һ�׵�
beta=0.2; %���׵�
gamma = 1; %��������
kappa = 0.1;  %
wl = 0; %ͼ������
we=50; %ͼ���ݶ�
wt=0;  %����
[row col] = size(Igs);

% ͼ����-�ߺ���
Eline = Igs;
% ͼ����-�ߺ���
[gx,gy]=gradient(Igs);
Eedge = -1*sqrt((gx.*gx+gy.*gy));
% ͼ����-�յ㺯��
% �����Ϊ�����ƫ����������ɢ���ƫ����������
m1 = [-1 1];
m2 = [-1;1];
m3 = [1 -2 1];
m4 = [1;-2;1];
m5 = [1 -1;-1 1];
cx = conv2(Igs,m1,'same');
cy = conv2(Igs,m2,'same');
cxx = conv2(Igs,m3,'same');
cyy = conv2(Igs,m4,'same');
cxy = conv2(Igs,m5,'same');
Eterm = (cyy.*cx.*cx -2 *cxy.*cx.*cy + cxx.*cy.*cy)./((1+cx.*cx + cy.*cy).^1.5);

%figure(3),imshow(Eterm);
%figure(4),imshow(abs(Eedge));
% �ⲿ�� Eext = Eimage + Econ
Eext = wl*Eline + we*Eedge + wt*Eterm;
% �����ݶ�
[fx,fy]=gradient(Eext);


m=11;
[mm nn] = size(fx);

% ������Խ�״����
% ��¼: ��ʽ��14�� b(i)��ʾviϵ��(i=i-2 �� i+2)
b(1)=beta;
b(2)=-(alpha + 4*beta);
b(3)=(2*alpha + 6 *beta);
b(4)=b(2);
b(5)=b(1);

B=(b'.*ones(5,m))';
A=full(spdiags(B,-2:2,m,m));
A(1,1:3)=0;A(2,4)=0;
A(end,end-2:end)=0;A(end-1,end-3)=0;
A(2,1:2:3)=-0.8;A(end-1,end-2:2:end)=-0.8;
% ����������
[L, U] = lu(A + gamma.* eye(m));
Ainv = inv(U) * inv(L);

% =========================================================================
%                      ��������
% =========================================================================

%points=[82,351,70,346;132,219,117,208;367,239,383,232;416,374,430,371];
[num_point,~]=size(points);c=2;

t=1:c;
ts = 1:0.1:c;
resx=[];resy=[];
for ii=1:num_point
    xy=[points(ii,1:2:3);points(ii,2:2:4)];
    
    xys = spline(t,xy,ts);
    xs = xys(1,:);
    ys = xys(2,:);
    xs=xs';
    ys=ys';
    
for i=1:NIter
    ssx = gamma*xs - kappa*interp2(fx,xs,ys);
    ssy = gamma*ys - kappa*interp2(fy,xs,ys);
    
    % ����snake����λ��
    xss = Ainv * ssx;
    yss = Ainv * ssy;
    if (sum((xs-xss).^2+(ys-yss).^2)/m)<1e-5
        break
    else
        xs=xss;ys=yss;
    end
    % ��ʾsnake����λ��
    if mod(i,10)==0
        imshow(Igs);
        hold on;
        plot(xs, ys, 'r-');
        title(['Iteration:',num2str(i)])
        hold off;
        pause(0.001)
    end
end
resx=[resx,xs];resy=[resy,ys];
end
show_img_contour(I, seg, 'r');
hold on;
for k=1:num_point
    plot(resx(:,k),resy(:,k), 'r-');
end
hold off;

end
