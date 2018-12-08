function MatingPool = F_mating(Population)
%�����ѡ��

    [N,D] = size(Population);
    
    %��Ԫ����ѡ��
    MatingPool = zeros(N,D);
    Rank = randperm(N);
    Pointer = 1;
    for i = 1 : 2 : N
        %ѡ��ĸ
        k = zeros(1,2);
        for j = 1 : 2
            if Pointer >= N
                Rank = randperm(N);
                Pointer = 1;
            end
            k = Rank(Pointer:Pointer+1);
            Pointer = Pointer+2;
        end
        MatingPool(i,:) = Population(k(1),:);
        MatingPool(i+1,:) = Population(k(2),:);
    end
end

