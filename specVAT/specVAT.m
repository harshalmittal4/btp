clc;
clear all;
close all;

neighbor_num = 6;         %% Number of neighbors to consider in local scaling
max_eigvalues = 11;

%%%%%%%%%%%%%%% load data 
%load Data6.mat;

%%%%%%%%%%%%%%% cluster all datasets
%X = XX{j};
     
    
%% centralize and scale the data
X = xlsread('data1.xlsx')
X = X - repmat(mean(X),size(X,1),1);
X = X/max(max(abs(X)));

figure;plot(X(:,1),X(:,2),'.')

%%%%%%%%%%%%%%%%% Compute local scale
D = distance2(X,X);              %% Euclidean distance

sigma=zeros(length(X),1);
for i=1:length(X)
    tmp=D(i,:);
    tmp(i)=[];
    tmp=sort(tmp);
    sigma(i)=tmp(neighbor_num);
end

%%%%%%%%%%%%%%%%% Construct the weighting matrix

W=zeros(length(X));

for i=1:length(X)
    for k=1:length(X)
        W(i,k)=exp((-1*D(i,k)*D(k,i))/(sigma(i)*sigma(k)));
    end
end
for i=1:length(X)
    W(i,i)=0;
end

%%%%%%%%%%%%%%%%% Construct the normalized version of the Laplacian matrix

M=zeros(length(X));
for i=1:length(X)
    M(i,i)=sum(W(i,:));
end
L_prime=M^(-0.5)*W*M^(-0.5);

%%%%%%%%%%%%%%%%% Find Eigenvectors and eigenvalues of L_prime

[V,D] = eig(L_prime);
[D,idx]=sort(diag(D),'descend');
V=V(:,idx);
gm = [];
for largest_eigvalues=1:max_eigvalues
    V_this=V(:,1:largest_eigvalues);

    %%%%%%%%%%%%%%%%% Normalize the rows of V
    % Matlab already retured unit norm eigenvectors, so this step is not needed
    % V_this_prime=zeros(length(X),largest_eigvalues);
    % for i=1:largest_eigvalues
    %     tmp=norm(V_this(:,i));
    %     V_this_prime(:,i)=V_this(:,i)/tmp;
    % end

    V_this_D=distance2(V_this,V_this);

    [RV,C,I,RI,cut]=VAT(V_this_D);
    [RiV,RV,reordering_mat]=iVAT(RV);
    %figure; imagesc(RiV); colormap(gray); axis image; axis off;
    %baseFileName_ = sprintf('ivat#%d.eps', largest_eigvalues);
    %fullFileName_ = fullfile(folder, baseFileName_);
    %print(fullFileName_,'-deps');

    %figure; imhist(RiV*256);
    %baseFileName = sprintf('hist#%d.eps', largest_eigvalues);
    %fullFileName = fullfile(folder, baseFileName);
    %print(fullFileName,'-deps');

    [counts_R,binsLocation_R]= imhist(RiV*256);
    %binsLocation_R = uint32(binsLocation_R*256)
    a = [1:256];
    b = reshape(a, [256, 1]);
    li = [];

    for t=1:256
        mu1 = sum(counts_R(1:t).*b(1:t))/sum(counts_R);
        mu2 = sum(counts_R(t+1:256).*b(t+1:256))/sum(counts_R);
        w1 = sum(counts_R(1:t))/sum(counts_R);
        w2 = sum(counts_R(t+1:256))/sum(counts_R);
        sigma_b = w1*w2*(mu1-mu2)*(mu1-mu2);
        li(end+1) = sigma_b;
    end
    [val, index] = max(li);          
    gm(end+1) = val;

end
[val_, index_] = max(gm);
%index_ is the best value of k determined by specVAT

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%generate and save the iVAT image and histogram for this k
V_this=V(:,1:index_);

V_this_D=distance2(V_this,V_this);

[RV,C,I,RI,cut]=VAT(V_this_D);
[RiV,RV,reordering_mat]=iVAT(RV);
figure; imagesc(RiV); colormap(gray); axis image; axis off;

folder = '/home/harshal/Desktop/Automatic cluster detection using machine learning/1'
%save to folder
baseFileName_ = sprintf('ivat#%d.eps', index_);
fullFileName_ = fullfile(folder, baseFileName_);
print(fullFileName_,'-deps');
        
figure; imhist(RiV*256);
%save to folder
baseFileName = sprintf('hist#%d.eps', index_);
fullFileName = fullfile(folder, baseFileName);
print(fullFileName,'-deps');
    
    
    













