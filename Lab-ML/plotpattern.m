function plotpattern(pat1,pat2)

%Visualizzazione di due pattern (binari) come immagini

for i=0:255
for j=1:3
map(i+1,:)=[i/255,i/255,i/255];
end
end
colormap(map);

buf1=zeros(13,8);
buf2=zeros(13,8);

count=1;
for j=1:13
        for k=1:8
            buf1(j,k)=pat1(1,count);
            buf2(j,k)=pat2(1,count);
            count=count+1;
        end
    end

count=1;
    subplot(1,2,count)
    image(255*(1-buf1));
    count=count+1;
    subplot(1,2,count)
    image(255*(1-buf2));
end
