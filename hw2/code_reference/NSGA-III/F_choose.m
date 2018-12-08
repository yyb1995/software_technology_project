function Choose = F_choose(FunctionValue1,FunctionValue2,K,Z)
%����ѡ��

    FunctionValue = [FunctionValue1;FunctionValue2];
    [N,M] = size(FunctionValue);
    N1 = size(FunctionValue1,1);
    N2 = size(FunctionValue2,1);
    NoZ = size(Z,1);

    %Ŀ�꺯��ֵ��һ��
    Zmin = min(FunctionValue,[],1);	%ÿά��Сֵ
    Extreme = zeros(1,M);           %ÿά�ı߽��
    w = zeros(M)+0.000001+eye(M);
    for i = 1 : M                   %�ҳ�ÿά�߽��
        [~,Extreme(i)] = min(max(FunctionValue./repmat(w(i,:),N,1),[],2));
    end
    Hyperplane = FunctionValue(Extreme,:)\ones(M,1);	%���㳬ƽ��
    a = 1./Hyperplane;             	%����ÿά�Ľؾ�
    if any(isnan(a))
        a = max(FunctionValue,[],1)';
    end
    FunctionValue = (FunctionValue-repmat(Zmin,N,1))./(repmat(a',N,1)-repmat(Zmin,N,1));	%��һ��
    
    %��ÿ�����������ĳ���ο���
    Distance = zeros(N,NoZ);        %����ÿ�����嵽ÿ���ο��������ľ���
    normZ = sum(Z.^2,2).^0.5;
    normF = sum(FunctionValue.^2,2).^0.5;
    for i = 1 : N
        normFZ = sum((repmat(FunctionValue(i,:),NoZ,1)-Z).^2,2).^0.5;
        for j = 1 : NoZ
            S1 = normF(i);
            S2 = normZ(j);
            S3 = normFZ(j);
            p = (S1+S2+S3)/2;
            Distance(i,j) = 2*sqrt(p*(p-S1)*(p-S2)*(p-S3))/S2;
        end
    end
    [d,pi] = min(Distance',[],1);   %��ÿ����������Ĳο�������
    
    %����ÿ���ο����������ĳ����һ����ĸ�����
    rho = zeros(1,NoZ);
    for i = 1 : N1
        rho(pi(i)) = rho(pi(i))+1;
    end
    
    %����ѡ��
    Choose = false(1,N2);       %����ѡ���ĸ�����
    Zchoose = true(1,NoZ);      %ʣ��δɾ���Ĳο���
    k = 1;
    while k <= K
        Temp = find(Zchoose);
        [~,j] = min(rho(Temp));
        j = Temp(j);
        I = find(Choose==0 & pi(N1+1:end)==j);
        if ~isempty(I)
            if rho(j) == 0
                [~,s] = min(d(N1+I));
            else
                s = randi([1,length(I)]);
            end
            Choose(I(s)) = true;
            rho(j) = rho(j)+1;
            k = k+1;
        else
            Zchoose(j) = false;
        end
    end
end

