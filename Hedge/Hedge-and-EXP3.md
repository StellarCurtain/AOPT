#  Hedge and Exp3 for sequential decision-making
- Written by Tommaso R. Cesari and Nicolo Cesa-Bianchi
- https://cesa-bianchi.di.unimi.it/Algo2/Note/hedge-exp3.pdf
# 问题描述
- 算法对一系列依次到达的请求做出相应
- 随机/对抗环境
- 存在多个专家给出预测
- 目标：最小化累计损失，使其接近最好专家的表现
# 基础问题设置:Prediction from expert advice
- 专家{1,…,K},可以被决策者和(对抗)环境选择
- 在每一轮t,环境秘密的为每个动作产生损失$\ell_t(i) \in [0,1]$
- 在每一轮t,决策者选择专家$I_t$,产生损失$\ell_t(I_t)$,并揭示所有专家的损失$\ell_t(i) \in [0,1],\ell_t \in [0,1]^K$
- 决策的评估:
  - 评估函数 $\mathbb{E}\left[ \frac{1}{T} \sum_{t=1}^{T} \ell_t(I_t) \right] - \min_{i=1,\dots,K} \left( \frac{1}{T} \sum_{t=1}^{T} \ell_t(i) \right)$
  - 目标: 找到合适的决策算法，使$T \to \infty$时，上式趋近于0
- 决策算法1：选择在过去表现最好的专家
  - $I_t = \arg \min_{i=1,\dots,K} \sum_{s=1}^{t-1} \ell_s(i)$
  - 缺陷:考虑每次为0的环境,专家1决策$\{0,1,0,1……\}$，专家2决策$\{\frac{1}{2},0,1,0……\}$，则每次都会基于之前的损失选择最差的专家，产生线性遗憾
- 决策算法2：Hedge
  - 参数:学习率$\gamma \in (0,1)$
  - 初始化:权重$w_1(i) = 1$
  - for t=1,2,…,T:
    - $W_t = \sum_{i=1}^{K} w_t(i)$,概率分布$p_t(i) = \frac{w_t(i)}{W_t}$
    - 根据概率分布$p_t$选择专家$I_t$
    - 产生损失$\ell_t(I_t)$,更新权重$w_{t+1}(i) = w_t(i) \cdot e^{-\gamma \ell_t(i)}$
    - $[w_{t+1}(i) = e^{-\gamma \sum_{s=1}^tl_s(i)}]$
- $
    \begin{aligned}
    \frac{W_{t+1}}{W_t} &= \sum_{i=1}^{K} \frac{w_{t+1}(i)}{W_t} 
    = \sum_{i=1}^{K} \frac{w_t(i) e^{-\gamma \ell_t(i)}}{W_t}
    = \sum_{i=1}^{K} p_t(i) e^{-\gamma \ell_t(i)} \\
    &\leq \sum_{i=1}^{K} p_t(i) \left( 1 - \gamma \ell_t(i) + \frac{\gamma^2}{2} \ell_t(i)^2 \right)\\
    &= 1 - \gamma \sum_{i=1}^{K} p_t(i) \ell_t(i) + \frac{\gamma^2}{2} \sum_{i=1}^{K}p_t(i) \ell_t(i)^2
    \end{aligned}\\
$
$\ln \frac{W_{t+1}}{W_t}\leq ln (1 - \gamma \sum_{i=1}^{K} p_t(i) \ell_t(i) + \frac{\gamma^2}{2} \sum_{i=1}^{K}p_t(i) \ell_t(i)^2)\\$

$\ln \left( \frac{W_T}{W_1} \right) = \sum_{t=1}^{T} \ln \left( \frac{W_{t+1}}{W_t} \right) \overset{\ln(1+z)\leq z}{\leq} -\gamma \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\ell_t(i) + \frac{\gamma^2}{2} \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\ell_t(i)^2\quad(1)$

$\ln \left( \frac{W_T+1}{W_1} \right) \overset{W_{t+1} = \sum_{i=1}^{K} w_{t+1}(i)}{\geq} \ln \left( \frac{w_{T+1}(k)}{W_1} \right) \overset{W_1=K}{=} -\gamma \sum_{t=1}^{T} \ell_t(k) - \ln(K)\quad(2)$

$(1)+(2):\sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\ell_t(i) - \sum_{t=1}^{T} \ell_t(k) \leq \frac{\ln K}{\gamma} + \frac{\gamma}{2} \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i) \ell_t(i)^2$

$\mathbb{E} \left[ \sum_{t=1}^{T} \ell_t(I_t) \right] - \sum_{t=1}^{T} \ell_t(k) \overset{p_i is distribution, l_t(i) \leq 1}{\leq} \frac{\ln K}{\gamma} + \frac{\gamma}{2} T\quad (3)$

对任意学习率$\gamma$与任意专家k成立,故取学习率$\gamma = \sqrt{\frac{2\ln K}{T}}$，则$\mathbb{E} \left[ \sum_{t=1}^{T} \ell_t(I_t) \right] - \sum_{t=1}^{T} \ell_t(k) \leq \sqrt{2T\ln K}$
考虑最优专家并同除$T:\mathbb{E} \left[ \frac{1}{T} \sum_{t=1}^{T} \ell_t(I_t) \right] - \min_{i=1,\dots,K} \left( \frac{1}{T} \sum_{t=1}^{T} \ell_t(i) \right) \leq \sqrt{\frac{2 \ln K}{T}}
$,即得目标函数的上界
- 有$\mathcal{O} (\sqrt{T})$的收敛速度
- 注:设置$w_{t+1}(i) = \exp \left( -\gamma_t \sum_{s=1}^{t} \ell_s(i) \right)$也可以得到相同的界

# 进阶问题设置:multi-armed bandit setting
- 每一轮仅揭示$l_t(I_t)$而非$l_t(i)$[仅揭示选择的专家的损失]
  - 考虑一个广告投放系统,仅知道选择的广告是否被点击,而不知道其他广告的点击情况
- 在每一轮t,环境秘密的为每个动作产生损失$\ell_t(i) \in [0,1]$
- 在每一轮t,决策者选择专家$I_t$,产生损失$\ell_t(I_t)$,并揭示它
- 评估函数 $\mathbb{E}\left[ \frac{1}{T} \sum_{t=1}^{T} \ell_t(I_t) \right] - \min_{i=1,\dots,K} \left( \frac{1}{T} \sum_{t=1}^{T} \ell_t(i) \right)$(与基础问题相同)
- EXP3算法
  - 用未揭示的损失$\ell_t(i)$的无偏估计值$\hat{\ell_t(i)}$来替换权重更新中的损失 $\ell_t(i)$
  - $\hat{\ell}_t(i) = \frac{\ell_t(i)}{p_t(i)} \mathbb{I}\{I_t = i\}$,其中$\hat{\ell}_t(i)$和$p_t(i)$是随机变量,$p_t(i)$由$I_1…I_{t-1}$决定
  - $\mathbb{E} \left[ \hat{\ell}_t(i) \mid I_1, \dots, I_{t-1} \right] = \mathbb{E} \left[ \frac{\ell_t(i)}{p_t(i)} \mathbb{I}\{I_t = i\} \mid I_1, \dots, I_{t-1} \right] = \frac{\ell_t(i)}{p_t(i)} p_t(i) = \ell_t(i)$(无偏估计)
  - $\mathbb{E} \left[ \hat{\ell}_t(i)^2 \mid I_1, \dots, I_{t-1} \right] = \mathbb{E} \left[ \frac{\ell_t(i)^2}{p_t(i)^2} \mathbb{I}\{I_t = i\} \mid I_1, \dots, I_{t-1} \right] = \frac{\ell_t(i)^2}{p_t(i)} \leq \frac{1}{p_t(i)}$
  - 同基础问题(1)+(2),可证$\sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\hat\ell_t(i) - \sum_{t=1}^{T} \hat\ell_t(k) \leq \frac{\ln K}{\gamma} + \frac{\gamma}{2} \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i) \hat\ell_t(i)^2$
  - $\mathbb{E} \left[ \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\hat{\ell}_t(i) \right] - \mathbb{E} \left[ \sum_{t=1}^{T} \ell_t(k) \right] \leq \frac{\ln K}{\gamma} + \frac{\gamma}{2} \mathbb{E} \left[ \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\hat{\ell}_t(i)^2 \right]$
  - $\mathbb{E} \left[ \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\mathbb{E} \left[ \hat{\ell}_t(i) \mid I_1, \dots, I_{t-1} \right] \right] - \mathbb{E} \left[ \sum_{t=1}^{T} \mathbb{E} \left[ \hat{\ell}_t(k) \mid I_1, \dots, I_{t-1} \right] \right] \leq \frac{\ln K}{\gamma} + \frac{\gamma}{2} \mathbb{E} \left[ \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\mathbb{E} \left[ \hat{\ell}_t(i)^2 \mid I_1, \dots, I_{t-1} \right] \right]$
  - $\mathbb{E} \left[ \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i)\ell_t(i) \right] - \sum_{t=1}^{T} \ell_t(k) \leq \frac{\ln K}{\gamma} + \frac{\gamma}{2} \mathbb{E} \left[ \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i) \frac{1}{p_t(i)} \right] \leq \frac{\ln K}{\gamma} + \frac{\gamma}{2}KT\quad (3)$
  - $\mathbb{E} \left[ \sum_{t=1}^{T} \sum_{i=1}^{K} p_t(i) \ell_t(i) \right] \overset{\mathbb{E}[\ell_t(I_t) \mid I_1, \dots, I_{t-1}] = \sum_{i=1}^{K} p_t(i) \ell_t(i)}{=} \mathbb{E} \left[ \sum_{t=1}^{T} \mathbb{E}[\ell_t(I_t) \mid I_1, \dots, I_{t-1}] \right] \overset{\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]}{=} \mathbb{E} \left[ \sum_{t=1}^{T} \ell_t(I_t) \right]$
  - 同除T,$\mathbb{E} \left[ \frac{1}{T} \sum_{t=1}^{T} \ell_t(I_t) \right] - \min_{i=1,\dots,K} \left( \frac{1}{T} \sum_{t=1}^{T} \ell_t(i) \right) \overset{\gamma = \sqrt{\frac{2\ln(K)}{KT}}}\leq \sqrt{\frac{2K \ln K}{T}}$
- 算法分析:
  - 有$\mathcal{O} (\sqrt{T})$的收敛速度
  - 仅多出$\sqrt{K}$的因子
  - 注:设置$w_{t+1}(i) = \exp \left( -\gamma_t \sum_{s=1}^{t} \ell_s(i) \right)$也可以得到相同的界

