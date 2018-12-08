# 北京航空航天大学软件技术基础作业一报告
------

<!-- TOC -->

- [北京航空航天大学软件技术基础作业一报告](#北京航空航天大学软件技术基础作业一报告)
    - [1 背景介绍](#1-背景介绍)
    - [2 仿真实验设计](#2-仿真实验设计)
        - [2.1 实验环境和初始条件](#21-实验环境和初始条件)
        - [2.2 随机事件对财富的作用](#22-随机事件对财富的作用)
        - [2.3 随机事件的移动](#23-随机事件的移动)
        - [2.4 其他仿真参数](#24-其他仿真参数)
    - [3 程序结构](#3-程序结构)
    - [4 仿真结果和分析](#4-仿真结果和分析)
        - [4.1 初始条件仿真结果](#41-初始条件仿真结果)
        - [4.2 总体财富分布](#42-总体财富分布)
        - [4.3 天赋与财富的关系](#43-天赋与财富的关系)
        - [4.4 最终财富分布与随机事件的关系](#44-最终财富分布与随机事件的关系)
    - [5 结论](#5-结论)
    - [6 参考文献](#6-参考文献)
    - [7 附录](#7-附录)

<!-- /TOC -->

## 1 背景介绍
在我们的传统观念中，一个人的成功主要取决于个人修为，例如智商、技能、天赋、努力程度和冒险精神等。而运气等外在因素有时只是锦上添花，只在一定程度上影响着财富的多少和社会地位的高低等。但是，最近arxiv上一篇论文通过一系列仿真实验，验证了一个在很多人看来有些不可思议的结论，那就是运气在一个人的成功中也扮演着不可忽视的作用。  
在经济学领域，存在着一条十分著名的定理：帕累托定理(Pareto's principle)。这条定理是19世纪末20世纪初意大利经济学家帕累托发现的。他认为，在任何一组东西中，最重要的只占其中一小部分，约20%，其余80%尽管是多数，却是次要的，因此又称二八定律。帕累托定理在社会财富的分布中同样适用。即20%的人占有着社会80%的财富，而80%的人只能获得剩下20%的财富。这条定理是100多年前被发现的，那么现在是否还适用呢？在[“全球贫富差距有多大？0.7%的人掌控着全球近一半财富”](https://wallstreetcn.com/articles/275107)这篇文章中，可以看在2016年的全球财富分布统计中，0.7%的人占有了46%的财富。这个结果说明如今人与人之间的财富差距越来越大。但是，我们知道，人类的先天智商分布是服从高斯分布的，中等智商的群体占了大多数。而在工作时间和工作效率方面确实存在差距。但是，人毕竟不是机器人，工作时间也是有上限的。那么，究竟是什么因素导致了如此巨大的贫富差距呢？在我看来，财富的差距是外部因素和个人修为共同作用的结果，而其中更重要的一个方面是外部因素，包括所处的国家地区，家庭条件，社会关系，各种机遇等。近年来，越来越多科学家认为一些成功或者不幸的原因只是因为简简单单的两个字：运气。最近的支付宝锦鲤也是运气的一个很好地体现。成为支付宝锦鲤的那位幸运儿，人生轨迹很可能因此而改变。她不需要再朝九晚五工作，只需要利用锦鲤带来的巨大流量和话题量就可以获得巨大的财富。互联网时代，一夜暴红不再是痴人说梦，只要你足够幸运。


## 2 仿真实验设计
本次仿真主要参考了文献[1]，设置了一个用于探究智商和随机事件对人群财富影响的实验。实验设计思路和参数设置如下：
### 2.1 实验环境和初始条件
将实验环境限定在一个201*201的方形区域内，以模拟实验人群的活动环境。在确定活动环境后，再定义实验人群和随机事件。为了模拟随机事件作用于实验人群上，在仿真过程中固定实验人群的位置，随机移动事件的位置。当随机事件进入实验人群的范围时，认为该实验个体遇到了随机事件。实验人群和随机事件初始分布均为均匀分布。为了研究智商与财富的关系，为实验人群设置了服从正态分布的智商，即
$$
X \sim N(\mu ,{\sigma ^2})\qquad1>X>0
$$
本次实验中，取$\mu=0.6$，$\sigma=0.1$。
### 2.2 随机事件对财富的作用
随机事件对分为两类：幸运事件和不幸事件。幸运事件对人群造成的影响是使实验人群的财富增长，增长的程度受智商的影响。具体的实现方式是：
设$c_t$为t时刻的财富，$p$为一个个体的智商，$p_0$为一个(0,1)的随机数。则该个体遇到幸运事件后，t+1时刻的财富为：
$$
c_{t+1}=
\begin{cases}
2c_t & p\ge{p_0} \\
c_t & p<{p_0}.
\end{cases}
$$
从上式可以看出，遇到幸运事件不代表着财富增长，财富是否增长与智商有关。智商越高，增长的概率越大。另一方面，不幸事件对人群的影响是使实验人群的财富减半，即：
$$
c_{t+1}=\frac{1}{2}c_t.
$$
如果这两种事件都没有遇到，则财富保持不变。
### 2.3 随机事件的移动
在仿真实验中，人群的位置是固定的，随机事件可以在区域内随机移动。移动方式为：设随机事件一次移动的距离为$l$，移动方向与x轴正半轴夹角为$\theta$，实验区域为$x_t\in{[0,201]},y_t\in[0,201]$。如果移动后的位置超出实验范围，则以试验边界为对称轴对称到实验区域内。随机实验的位置$(x_t,y_t)$的变化为：
$$
x_{t+1}=
\begin{cases}
402-({x_t+l*cos\theta}) & {x_t+l*cos\theta}\ge201 \\
\left|{x_t+l*cos\theta}\right| & \left|{x_t+l*cos\theta}\right|<201 \\
\end{cases}
\\\\
y_{t+1}=
\begin{cases}
402-({y_t+l*sin\theta}) & {y_t+l*sin\theta}\ge201 \\
\left|{y_t+l*sin\theta}\right| & \left|{y_t+l*sin\theta}\right|<201
\end{cases}
$$
### 2.4 其他仿真参数
本实验使用的其他参数如下表所示：

|参数名称|参数值|
| :-:| :-: |
| 实验个体总数| 1000|
| 随机事件总数 | 1000 |
|幸运事件比例|50%|
|随机事件移动距离|2|
|仿真总时间|40|
|随机事件移动时间间隔|0.5|
## 3 程序结构
本次仿真实验程序使用python语言编写，版本为python3.6.2。实验程序主要包括3个文件：`main.py`、`utils.py`和`draw_result.py`。`main.py`是主程序，用于实现整个仿真流程，`utils.py`用于对实验条件进行初始化，`draw_result.py`用于对实验数据进行处理和图片展示。

`main.py`中主要使用以下两个类来完成需求：Individual类和Incident类
1. Individual类是实验个体类，包含的属性有姓名(name)，智商(talent)，财富(capital)，位置(location)，事件记录(capital_record)等。包含的方法有`encounter_incident()`和`get_full_incident_record()`。前者用于判断在某一时刻这个个体是否遇到随机事件以及计算财富变化。后者用于在仿真实验完成后生成这个个体的完整财富记录，用于结果分析。
2. Incident类是随机事件类。包含的属性有序号(name)，时刻(time)，位置(location)和属性(islucky)。方法有`move()`。这个方法生成下一时刻该事件的位置。

```python
class Individual:
    # Implement Individual class
    def __init__(self, name, talent, init_capital, location):
        self.name = name
        self.talent = float(talent)
        self.capital = float(init_capital)
        self.location = location
        self.incident_record = []
        self.full_incident_record = []
        self.lucky_incident_num = 0
        self.unlucky_incident_num = 0

    def encounter_incident(self, incident_time, incident_location, is_lucky, talent_boundary):
        """
        Update the capital according to possible incident. Encounter means distance between individual
        and incident is <= 1. incident_record is a list. Each element is a list:[incident_time, flag, capital].
        flag is a signal of incident type. 1 indicates one gain capital, 0 indicates no changes,
        -1 indicates one lose capital.
        :param incident_time: Incident happen time
        :param incident_location: Current incident location
        :param is_lucky: Boolean. If lucky, the value is True, else is False
        :param talent_boundary: Judge whether an individual can benefit from an incident
        :return:
        """
        if np.linalg.norm(self.location - incident_location) <= 1:
            if is_lucky:
                # talent_boundary is the lower boundary that an individual can benefit from a lucky incident
                if self.talent > talent_boundary:
                    self.capital = self.capital * 2
                    self.incident_record.append([incident_time, 1, self.capital])
                    self.lucky_incident_num = self.lucky_incident_num + 1
            else:
                self.capital = self.capital / 2
                self.incident_record.append([incident_time, -1, self.capital])
                self.unlucky_incident_num = self.unlucky_incident_num + 1

    def get_full_incident_record(self, incident_num, init_capital):
        """
        Generate the full incident and capital record of an individual.
        :param incident_num: total incident num, should be equal to sim_time / time_interval
        :param init_capital:initial capital
        :return: full_incident_record. First col is incident number. Second column is incident type. 0 means
        no incident. 1 means lucky incident. -1 means unlucky incident. Third column is capital record.
        """
        full_incident_record = np.zeros((incident_num, 3), dtype=float)
        full_incident_record[:, 2] = init_capital * np.ones(full_incident_record.shape[0], dtype=float)
        full_incident_record[:, 0] = np.array([(i + 1) for i in range(1, incident_num + 1)])
        # fill the full record with list record.
        for record in self.incident_record:
            full_incident_record[record[0] - 1, :] = record
        for i in range(1, incident_num):
            # no incident happens to an individual. his capital remains same.
            if np.abs(full_incident_record[i, 2] - 10) < 1e-10:
                full_incident_record[i, 2] = full_incident_record[i - 1, 2]
        self.full_incident_record = full_incident_record
        return full_incident_record


class Incident:
    # Implement Incident class
    def __init__(self, name, init_location, islucky):
        self.name = name
        self.time = 0
        self.location = init_location
        self.islucky = islucky

    def move(self, length, direction, max_x_boundary, max_y_boundary):
        # Generate next location of this incident. If the location i整个s out of range, use that axis as a symmetry axis
        # and reflect it into the range. Minimum boundary is set to 0 by default.
        if self.location[0] + length * np.cos(direction * np.pi / 180) > max_x_boundary:
            self.location[0] = 2 * max_x_boundary - (self.location[0] + length * np.cos(direction * np.pi / 180))
        elif self.location[1] + length * np.sin(direction * np.pi / 180) > max_y_boundary:
            self.location[1] = 2 * max_y_boundary - (self.location[1] + length * np.sin(direction * np.pi / 180))
        else:
            self.location = np.abs([self.location[0] + length * np.cos(direction * np.pi / 180),
                                    self.location[1] + length * np.sin(direction * np.pi / 180)])

```

仿真实验包括单次仿真和多次仿真。单次仿真对应`main.py`中的`single_run()`函数。在单次仿真流程中，首先调用`utils.py`的`generate_distribution()`和`generate_location()`函数生成包含所有实验个体和随机事件的列表。在总仿真时间`sim_time`内，每隔一个时间间隔`time_interval`，先调用所有随机事件的`move()`方法，然后依次调用每个实验个体的`encounter_incident()`方法，判断该个体是否遇上了随机事件并记录财富变化。当整个仿真完成后，保存仿真结果，并调用`draw_result.py`中的不同函数展示仿真结果。多次仿真对应`multiple_run()`函数。


## 4 仿真结果和分析
### 4.1 初始条件仿真结果
在初始条件中，包括初始实验人群和随机事件分布以及智商分布。这两个仿真结果如下两图所示。

*初始实验个体和随机事件位置分布*

![](https://raw.githubusercontent.com/yyb1995/software_technology_project/release/hw1/result/initial_location.png)

*初始天赋分布*

![](https://raw.githubusercontent.com/yyb1995/software_technology_project/release/hw1/result/static_talent.png)

从实验条件得知，实验人群和随机事件位置分布服从均匀分布，初始天赋分布服从正态分布。从图中可以看出，初始位置分布近似均匀分布，初始天赋分布符合均值为0.6，方差为0.1的正态分布，且最大值不大于1，最小值不小于0。

### 4.2 总体财富分布
*各财富段人数分布*

![](https://raw.githubusercontent.com/yyb1995/software_technology_project/release/hw1/result/static_capital_individual_num.png)
上图反映的是在经过一次完整的仿真流程后各财富段的人数分布。从第一个分图可以直观地看出贫富分布不均。绝大多数人只拥有很少一部分财富，而极少数人却拥有着很大一部分财富。这与第一部分中介绍的现实世界的财富帕累托定律比较吻合。为了进一步揭示财富分布与人数的关系，使用帕累托分布对实验结果进行拟合。拟合结果如第二个分图所示。从图中可以看出拟合效果较好，拟合结果是$y(c)\sim{c^{-0.965}}$。此外，还统计了财富和人数的比例，得到的结果是最富有的20%的人拥有82%的财富。剩下的80%的人只拥有18%的财富。这个结果与帕累托定理比较吻合。这个结果说明即使实验人群的初始条件服从正态分布，但是在经历一系列不同的事件后，财富的分布差异变得非常大。
### 4.3 天赋与财富的关系
*最终财富分布与天赋的关系*

![](https://raw.githubusercontent.com/yyb1995/software_technology_project/release/hw1/result/static_capital_talent.png) 
从两个角度分析上图。第一：总体上看，财富较多的群体更多地分布在智商在均值之上的群体中。这个结论说明较高的智商有助于从幸运事件中获利。第二：拥有财富最多的个体智商仅在0.7附近，而智商接近0.9的个体也并没有拥有很多财富。从以上现象可以得出一个结论：智商并不是决定财富数量的一个决定性因素。

### 4.4 最终财富分布与随机事件的关系
*最终财富分布与随机事件数量的关系*

![](https://raw.githubusercontent.com/yyb1995/software_technology_project/release/hw1/result/static_capital_incident.png)
在本节中对实验个体最终财富与随机事件数量的关系进行分析。从第一个分图可以看出，遇到的幸运事件中最多的实验个体同时也是拥有最多财富的个体。遇到的幸运事件比较多的个体拥有的财富也接近最多。从第二个分图可以看出，遇到不幸事件较多的个体的财富接近于0。这个结论说明了遇到幸运事件的多少从很大程度上决定了一个人获得巨大财富的可能性。遇到的幸运事件多，即使天赋并不出众也能获得巨大的财富收益；几乎没有遇到幸运事件或遇到的不幸事件多，即使天赋出众也难以获得很多财富。此外，从图中还可以看出，绝大多数实验个体在整个仿真时间段内并没有遭遇过很多幸运事件和不幸事件，这使得大部分实验个体的财富一直比较少。下面两图具体展示了最富有个体和最贫穷个体的财富变化。

*最富有的个体财富变化*

![](https://raw.githubusercontent.com/yyb1995/software_technology_project/release/hw1/result/richest_individual_record.png)

*最贫穷个体财富变化*

![](https://raw.githubusercontent.com/yyb1995/software_technology_project/release/hw1/result/poorest_individual_record.png)
从以上两图可以观察到，最富有个体在仿真周期中一共遇上了9次幸运事件，并且都成功地把握住了增长财富的机会。而最贫穷个体在仿真周期中一共遇上了14次不幸事件。这使他的财富在整个仿真过程中一再缩水。由于一直遭遇着不幸事件，因此他的天赋并没有帮助他增长财富。


## 5 结论
通过对以上实验结果的分析，可以得出以下初步结论：
   1. 智商并不是影响一个人获得财富的决定性因素
   2. 幸运程度很大程度上决定了一个人最终能取得财富的多少

我觉得实验结果给我们普通人最大的启发是：我们不必过于羡慕那些天资聪颖的神童。因为，天赋仅仅是成功的一小块拼图而已。此外，实验结论与我们经常听说的一句名言：*天才是由99%的汗水和1%的聪慧组成的* 有些矛盾。这次实验目前暂时没有考虑努力这一因素，但是从现有的结果来看，幸运这一要素对于成功的贡献不可忽视。但是，我想强调的一点是，无论是天赋还是幸运，一定程度上都属于外部因素，是我们很难改变或预见的。那是否应该听天由命，由上天为我们安排人生呢？答案显然是否定的。我们应该充分发挥出人的主观能动性，也就是通过学习和修炼，不断创造出属于自己的财富，提高社会地位，增大努力这一因素在成功里的分量。而且，正如一句话说的那样：*越努力越幸运* 。努力和幸运其实是存在正相关关系的。只有自己准备好了，才能在幸运到来时紧紧地将它抓住。最后，也许努力并不能让我们成为锦鲤或The Chosen One，但努力毫无疑问会让我们活的比现在更好!

## 6 参考文献
[[1] Pluchino A, Biondo A E, Rapisarda A. TALENT VERSUS LUCK: THE ROLE OF RANDOMNESS IN SUCCESS AND FAILURE[J]. 2018.](https://arxiv.org/abs/1802.07068)

## 7 附录
程序代码和仿真结果：[https://github.com/yyb1995/software_technology_project](https://github.com/yyb1995/software_technology_project)


