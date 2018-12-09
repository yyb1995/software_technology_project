# 北京航空航天大学软件技术基础作业二报告
------



## 1 背景介绍
NSGA(Non-dominated sorting genetic algorithm)算法可以用来解决单目标或多目标的优化问题。在多目标优化问题中，通常不能得到单个最优解，取而代之的是一系列非支配解，称为帕累托最优解(Pareto-optimal solutions)或非支配解(nondominated soluitons)。这些解的特点是：无法在改进任何目标函数与的同时不削弱至少一个其他目标函数。帕累托最优解的定义为：
>对于最小化多目标问题，n个目标分量$f_i(i=1, 2, \cdots,n)$组成的向量$\overline{f}(\overline{X})=(f_1(\overline{X}, f_2(\overline{X}),\cdots, f_n(\overline{X})$，$\overline{X}\in{U}$为决策变量。若$\overline{X}\in{U}$为Pareto最优解，则满足：
不存在决策变量$\overline{X}_{i}, \overline{X}_{u}\in{U}$使得下式成立：
$$
\forall{i}\in{\{1,2,\cdots,n\}}, f(\overline{X}_{i})\le{f(\overline{X}_{u})}
$$

Pareto解的集合即所谓的帕累托前沿(Pareto Front)。在Pareto front中的所有解皆不受Pareto Front之外的解（以及Pareto Front 曲线以内的其它解）所支配，因此这些非支配解较其他解而言拥有最少的目标冲突，可提供决策者一个较佳的选择空间。在某个非支配解的基础上改进任何目标函数的同时，必然会削弱至少一个其他目标函数。而各种多目标优化算法的目标就是在迭代次数或精度要求内更好地逼近Pareto Front。

NSGA系列算法与简单的遗传算法的主要区别在于：该算法在选择算子执行之前根据个体之间的支配关系进行了分层。其选择算子、交叉算子和变异算子和简单的遗传算法没有区别。非支配分层方法可以使好的个体有更大的机会遗传到下一代；适应度共享策略则使得准Pareto面上的个体均匀分布，保持了群体的多样性，克服了超级个体的过度繁殖，防止了早熟收敛。

下图是NSGA2算法的求解流程，NGSA3算法流程与之类似。两种算法的最大区别在于非支配个体的选择。NSGA2用拥挤距离对同一非支配等级的个体进行选择（拥挤距离越大越好），而NSGA3用的是基于参考点的方法对个体进行选择。NSGA3采用基于参考点的方法就是为了解决在面对三个及其以上目标的多目标优化问题时，如果继续采用拥挤距离的话，算法的收敛性和多样性不好的问题（就是得到的解在非支配层上分布不均匀，这样会导致算法陷入局部最优）。
![](https://images2015.cnblogs.com/blog/65150/201603/65150-20160312144949460-644627639.png)


## 2 仿真实验设计
本次仿真主要参考了文献[1]，对NSGA3算法进行了实现和测试。实验设计思路和参数设置如下：
### 2.1 实验环境及参数设置
本次实验使用python3.6实现NSGA3算法。主要参考的包括密歇根州立大学colab实验室公布的pymoo和pymop。pymoo中实现了多种多目标优化算法，包括NSGA2，NSGA3，R-NSGA3等。pymop包中给出了多种多目标测试函数，包括ZDT、DTLZ系列等。实验中设置的一些参数如下表所示：

|参数名|参数值|
|--|--|
|种群规模|100|
|迭代次数|100|
|随机数种子|5|

### 2.2 测试函数
为了测试NSGA算法的性能，本次实验选取了以下测试函数
1. 自定义测试函数
$$ 
f_1(x_1, x_2)={x_1}^4-10{x_1}^2+x_1x_2+{x_2}^4-{x_1}^2{x_2}^2\\
f_2(x_1, x_2)={x_1}^4+{x_2}^4+x_1x_2-{x_1}^2{x_2}^2\\
s.t.\quad \begin{cases}
5\ge x_1\ge-5 \\
5\ge x_2\ge-5
\end{cases}
$$

2. Osyczka and Kundu function
$$
f_1(x) = -25(x_1-2)^2-(x_2-2)^2-(x_3-1)^2-(x_4-4)^2-(x_5-1)^2\\
f_2(x)=\sum\nolimits_{n=1}^{6}x_i^2\\
s.t. \quad \begin{cases}
6\ge x_1+x_2\ge2 \\1. 自定义测试函数
$$ 
f_1(x_1, x_2)={x_1}^4-10{x_1}^2+x_1x_2+{x_2}^4-{x_1}^2{x_2}^2\\
f_2(x_1, x_2)={x_1}^4+{x_2}^4+x_1x_2-{x_1}^2{x_2}^2\\
s.t.\quad \begin{cases}
5\ge x_1\ge-5 \\
5\ge x_2\ge-5
\end{cases}
$$

2. Osyczka and Kundu function
$$
f_1(x) = -25(x_1-2)^2-(x_2-2)^2-(x_3-1)^2-(x_4-4)^2-(x_5-1)^2\\
f_2(x)=\sum\nolimits_{n=1}^{6}x_i^2\\
s.t. \quad \begin{cases}
6\ge x_1+x_2\ge2 \\
x_1-x_2\ge-2\\
-x_1+3x_2\ge-2\\
4-(x_3-3)^2-x_4\ge0\\
(x_5-3)^2+x_6-4\ge0
\end{cases}
$$

3. DTLZ function
$$
f_1(x)=\frac{1}{2}x_1(1+g(\vec x))\\
f_2(x)=\frac{1}{2}(1-x_1)(1+g(\vec x))
g(\vec x)=100\left\{\left|\vec x\right|+\sum_{x_i\in{\vec x}}(x_1-0.5)^2-\cos\left[20\cdot \pi (x_i-0.5)\right]\right\}\\
s.t.\quad 1\ge x_i\ge0,i=1,\cdots n
$$

4. Viennet function
x_1-x_2\ge-2\\
-x_1+3x_2\ge-2\\
4-(x_3-3)^2-x_4\ge0\\
(x_5-3)^2+x_6-4\ge0
\end{cases}
$$

3. DTLZ1 function
$$
f_1(x)=\frac{1}{2}x_1(1+g(\vec x))\\
f_2(x)=\frac{1}{2}(1-x_1)(1+g(\vec x))
g(\vec x)=100\left\{\left|\vec x\right|+\sum_{x_i\in{\vec x}}(x_1-0.5)^2-\cos\left[20\cdot \pi (x_i-0.5)\right]\right\}\\
s.t.\quad 1\ge x_i\ge0,i=1,\cdots n
$$

4. Viennet function
$$
f_1(x_1, x_2)=0.5(x_1^2+x_2^2)+\sin(x_1^2+x_2^2)\\
f_2(x_1, x_2)=\frac{(3x_1-2x_2+4)^2}{8}+\frac{(x_1-x_2+1)^2}{27}+15\\
f_3(x_1, x_2)=\frac{1}{(x_1^2+x_2^2+1)^2}-1.1\exp{-(x_1^2+x_2^2)}\\
s.t.\quad 3\ge x_1, x_2\ge-3
$$


## 3 程序结构
本次实验的主程序为`main.py`。其中包括两个函数：`use_custom_function`和`use_package_function`以及一个类`Myproblem`。`Myproblem`中的`_evaluate()`方法用于自定义待求解函数。`use_custom_function`用于求解自定义函数，`use_package_function`用于求解pymop包中实现的待求解函数。一次完整的求解流程为：
1. 调用`UniformReferenceDirectionDactroy`函数得到种群的初始值分布。在本实验中，初始分布为(0,1)之间的均匀分布。
2. 在`minimize`函数中，调用`solve`方法进行问题求解。`solve`方法实际上是一个类方法，在不同的优化算法中有不同的实现。
```python
    def solve(self,
              problem,
              termination,
              seed=1,
              disp=False,
              callback=None,
              save_history=False,
              pf=None,
              **kwargs
              ):
        """

        Solve a given problem by a given evaluator. The evaluator determines the
        termination condition and can either have a maximum budget, hypervolume
        or whatever. The problem can be any problem the algorithm is able to
        solve.

        Parameters
        ----------

        problem: class
            Problem to be solved by the algorithm

        termination: class
            object that evaluates and saves the number of evaluations and
            determines the stopping condition

        seed: int
            Random seed for this run. Before the algorithm starts this seed is
            set.

        disp : bool
            If it is true that information during the algorithm execution are
            displayed.

        callback : func
            A callback function can be passed that is executed every generation.
            The parameters for the function are the algorithm itself, the number
            of evaluations so far and the current population.

                def callback(algorithm):
                    pass

        save_history : bool
            If true, a current snapshot of each generation is saved.

        pf : np.array
            The Pareto-front for the given problem. If provided performance
            metrics are printed during execution.

        Returns
        -------
        res : dict
            A dictionary that saves all the results of the algorithm. Also,
            the history if save_history is true.

        """

        # set the random seed for generator
        random.seed(seed)

        # the evaluator object which is counting the evaluations
        self.evaluator = Evaluator()
        self.problem = problem
        self.termination = termination
        self.pf = pf

        self.disp = disp
        self.callback = callback
        self.save_history = save_history

        # call the algorithm to solve the problem
        pop = self._solve(problem, termination)

        # get the optimal result by filtering feasible and non-dominated
        opt = pop.copy()
        opt = opt[opt.collect(lambda ind: ind.feasible)[:, 0]]

        # if at least one feasible solution was found
        if len(opt) > 0:

            if problem.n_obj > 1:
                I = NonDominatedSorting().do(opt.get("F"),
                                             only_non_dominated_front=True)
                opt = opt[I]
                X, F, CV, G = opt.get("X", "F", "CV", "G")

            else:
                opt = pop[np.argmin(pop.get("F"))]
                X, F, CV, G = opt.X, opt.F, opt.CV, opt.G
        else:
            opt = None

        res = Result(opt, opt is None, "")
        res.algorithm, res.problem, res.pf = self, problem, pf
        res.pop = pop

        if opt is not None:
            res.X, res.F, res.CV, res.G = X, F, CV, G

        res.history = self.history

        return res
```

3. 在NSGA3的求解实现中，模型各部分封装成类并放在`./pymoo/operators/`中。在每一次迭代中，先对种群进行交叉：`SimulatedBinaryCrossover`和多项式变异：`PolynomialMutation`。接着需要选择留下的种群并进行重新组合加入下一次迭代。选择的函数为`ReferenceDirectionSurvival.do()`。该函数首先对种群进行非支配排序：`NonDominatedSorting.do()`。在排序后，保留前面的非支配层，从最后一个非支配排序层选择一些个体加入下一次迭代直到数量达到留下数量上限。选择的方法是通过`get_extreme_points_c`得到极值点，再通过`get_nadir_point`得到截距，再通过`associate_to_niches`进行归一化，最后通过`calc_niche_count`得到留下的点。`ReferenceDirectionSurvival.do()`的代码如下：

```python 
def _do(self, pop, n_survive, D=None, **kwargs):

    # attributes to be set after the survival
    F = pop.get("F")

    # find or usually update the new ideal point - from feasible solutions
    self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
    self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

    # calculate the fronts of the population
    fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
    non_dominated, last_front = fronts[0], fronts[-1]

    # find the extreme points for normalization
    self.extreme_points = get_extreme_points_c(F[non_dominated, :], self.ideal_point,
                                                extreme_points=self.extreme_points)

    # find the intercepts for normalization and do backup if gaussian elimination fails
    worst_of_population = np.max(F, axis=0)
    worst_of_front = np.max(F[non_dominated, :], axis=0)

    self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                        worst_of_population, worst_of_front)

    #  consider only the population until we come to the splitting front
    I = np.concatenate(fronts)
    pop, rank, F = pop[I], rank[I], F[I]

    # update the front indices for the current population
    counter = 0
    for i in range(len(fronts)):
        for j in range(len(fronts[i])):
            fronts[i][j] = counter
            counter += 1
    last_front = fronts[-1]

    # associate individuals to niches
    niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point, self.nadir_point)
    pop.set('rank', rank, 'niche', niche_of_individuals, 'dist_to_niche', dist_to_niche)

    # if we need to select individuals to survive
    if len(pop) > n_survive:

        # if there is only one front
        if len(fronts) == 1:
            n_remaining = n_survive
            until_last_front = np.array([], dtype=np.int)
            niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

        # if some individuals already survived
        else:
            until_last_front = np.concatenate(fronts[:-1])
            niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
            n_remaining = n_survive - len(until_last_front)

        S = niching(F[last_front, :], n_remaining, niche_count, niche_of_individuals[last_front],
                    dist_to_niche[last_front])

        survivors = np.concatenate((until_last_front, last_front[S].tolist()))
        pop = pop[survivors]

    return pop
```
4. 仿真的Pareto Front结果图通过`plotting.plot()`函数进行展示和保存。


## 4 仿真结果和分析
### 4.1 自定义测试函数仿真结果
![]()

### 4.2 Osyczka and Kundu function仿真结果

*参考结果：*
![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Osyczka_and_Kundu_function.pdf/page1-1194px-Osyczka_and_Kundu_function.pdf.jpg)
### 4.3 DTLZ1 function仿真结果

![](http://delta.cs.cinvestav.mx/~ccoello/EMOO/testfuncs/mopfigs/dtlz1funb.jpg)



### 4.4 Viennet function仿真结果


*参考结果：*
![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Viennet_function.pdf/page1-1194px-Viennet_function.pdf.jpg)



## 5 结论
通过对以上实验结果的分析，可以得出以下初步结论：
   1. 智商并不是影响一个人获得财富的决定性因素
   2. 幸运程度很大程度上决定了一个人最终能取得财富的多少

我觉得实验结果给我们普通人最大的启发是：我们不必过于羡慕那些天资聪颖的神童。因为，天赋仅仅是成功的一小块拼图而已。此外，实验结论与我们经常听说的一句名言：*天才是由99%的汗水和1%的聪慧组成的* 有些矛盾。这次实验目前暂时没有考虑努力这一因素，但是从现有的结果来看，幸运这一要素对于成功的贡献不可忽视。但是，我想强调的一点是，无论是天赋还是幸运，一定程度上都属于外部因素，是我们很难改变或预见的。那是否应该听天由命，由上天为我们安排人生呢？答案显然是否定的。我们应该充分发挥出人的主观能动性，也就是通过学习和修炼，不断创造出属于自己的财富，提高社会地位，增大努力这一因素在成功里的分量。而且，正如一句话说的那样：*越努力越幸运* 。努力和幸运其实是存在正相关关系的。只有自己准备好了，才能在幸运到来时紧紧地将它抓住。最后，也许努力并不能让我们成为锦鲤或The Chosen One，但努力毫无疑问会让我们活的比现在更好!

## 6 参考文献及资料
[[1] Deb K, Jain H. An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints[J]. IEEE Transactions on Evolutionary Computation, 2014, 18(4):577-601.](https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf)

[[2] Deb K, Pratap A, Agarwal S, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(2):182-197.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=996017)

[[3] NSGA-II explained!](http://oklahomaanalytics.com/data-science-techniques/nsga-ii-explained/)

[[4] Single- as well as Multi-Objective Optimization Test Problems: ZDT, DTLZ, WFG, BNH, OSY, ...](https://github.com/msu-coinlab/pymop)

[[5] NSGA2, NSGA3, R-NSGA3, MOEAD, GA, DE,](https://github.com/msu-coinlab/pymoo)
## 7 附录
程序代码和仿真结果：[https://github.com/yyb1995/software_technology_project](https://github.com/yyb1995/software_technology_project)


