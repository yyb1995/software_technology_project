%NSGA-III
function MAIN(Problem,M,Run)
clc;format compact;tic;
%-----------------------------------------------------------------------------------------
%�����趨
    [Generations,N,p1,p2] = P_settings('NSGA-III',Problem,M);
%-----------------------------------------------------------------------------------------      
%�㷨��ʼ
    %��ʼ������
    Evaluations = Generations*N;
    [N,Z] = F_weight(p1,p2,M);
    Z(Z==0) = 0.000001;
    Generations = floor(Evaluations/N);
    
    %��ʼ����Ⱥ
    [Population,Boundary,Coding] = P_objective('init',Problem,M,N);
    
    %��ʼ����
    for Gene = 1 : Generations 
        %�����Ӵ�
        MatingPool = F_mating(Population);
        Offspring = P_generator(MatingPool,Boundary,Coding,N);
        Population = [Population;Offspring];
        FunctionValue = P_objective('value',Problem,M,Population);

        [FrontValue,MaxFront] = P_sort(FunctionValue,'half');

        %ѡ����֧��ĸ���        
        Next = zeros(1,N);
        NoN = numel(FrontValue,FrontValue<MaxFront);
        Next(1:NoN) = find(FrontValue<MaxFront);
        
        %ѡ�����һ����ĸ���
        Last = find(FrontValue==MaxFront);
        Choose = F_choose(FunctionValue(Next(1:NoN),:),FunctionValue(Last,:),N-NoN,Z);
        Next(NoN+1:N) = Last(Choose);
        
        %��һ����Ⱥ
        Population = Population(Next,:);
        
%         F = P_objective('value',Problem,M,Population);
%         cla;
%         P_draw(F);
%         pause(0.2);
        clc;fprintf('NSGA-III,��%2s��,%5s����,��%2sά,�����%4s%%,��ʱ%5s��\n',num2str(Run),Problem,num2str(M),num2str(round2(Gene/Generations*100,-1)),num2str(round2(toc,-2)));
    end
%-----------------------------------------------------------------------------------------     
%���ɽ��
    P_output(Population,toc,'NSGA-III',Problem,M,Run);
end