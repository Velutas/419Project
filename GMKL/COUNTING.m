load Temp

Data = 1200

x = zeros(1,Data)
ToCount = TestPos;

for j = 1:Data
    for i = 1:1280
        if ToCount(i,j) >= 1
            x(j) = x(j) + ToCount(i,j)
        end
    end
end
