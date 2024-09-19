function dataOut = generate_shifted_data(dataIn, shift)
% data = generate_shifted_data(data, shift)

if size(dataIn,3)>1
    dataIn=permute(dataIn, [1 3 2]);
end
Ls=length(shift);
dataOut=dataIn;

for k=1:Ls
    if shift(k)
    dataOut = [dataOut circshift(dataIn,shift(k))];
    end
end

if size(dataOut,3)>1
    dataOut=permute(dataOut, [1 3 2]);
end
