# Adversarial Bandits

# Part 1. Bandits

## 1.1 Bandit Problems
- Also called partial-information online learning

在线学习的一般形式:
- At each round $t=1,2,\cdots$
- - (1) player 选择model $w_t \in W$
- - (2) simulaneously environment 选择online function $f_t : W \to \mathbb{R}$
- - (3) player 受到$f_t(w_t)$的损失,获取$f_t$的信息并更新模型
- - bandits: 在(3)中仅揭示$f_t(w_t)$(该点函数值)

## 1.2 Adversarial Bandits 
- 本课程针对oblivious setting(环境不会针对learner)
- ![alt text](image.png)

# Part 2. (Adversarial) Multi-Armed Bandits

## 2.1 Formulation
- ![alt text](image-1.png)
- ![alt text](image-2.png)
- ![alt text](image-3.png)
## 2.2 Loss Estimator


## 2.3 Exp3 and Regret Analysis












