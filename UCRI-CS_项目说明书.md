---
title: "UCRI-CS 项目说明书"
subtitle: "面向选择偏差、拒贷无标签、不可观测选择偏差与决策校准的半监督信用评分框架（v8）"
author: "项目预研文档"
date: "2026-06-06（v8 修订版）"
lang: zh-CN
---

# UCRI-CS 项目说明书（v8 修订版）

**项目全称：** UCRI-CS: Uncertainty-Calibrated Semi-Supervised Reject Inference for Decision-Aware Credit Scoring  
**中文名称：** 面向信贷决策的半监督拒贷推断与不确定性校准信用评分框架  
**建议论文题目：** Uncertainty-Calibrated Semi-Supervised Reject Inference for Decision-Aware Credit Scoring  
**目标投稿定位：** CCF B/B+ 数据挖掘、AI 应用、金融科技、信息系统方向会议/期刊；优先考虑 CIKM、ICDM、ECML-PKDD、DASFAA、PAKDD、KAIS、Information Sciences、Expert Systems with Applications、Knowledge-Based Systems 等。  
**核心任务：** 在只有已放款样本有真实违约标签、拒贷样本无还款结果标签的场景下，学习可校准、可解释、可用于审批决策的违约概率估计模型。

**本 v8 修订版重点修正：**

```text
1. 显式补充 reject inference 的可识别性假设与适用边界；
2. 对 LendingClub rejected 数据中的 Risk_Score 设置循环依赖审计；
3. 重构 simulated rejection protocol，避免模拟机制与模型机制同构导致循环验证；
4. 明确真实 rejected 样本无直接标签的验证局限；
5. 增加 PU learning baseline、类别不平衡处理、额外 accepted 数据对照和 Risk_Score-only baseline；
6. 显式处理 teacher 在 rejected 分布上的 covariate shift 与 cross-population calibration；
7. 增加 confounded rejection simulation，测试 conditional ignorability 不完全成立时的性能退化；
8. 固定 ECE 分箱、标签定义敏感性、τ_u 阈值敏感性、ensemble 成员数消融和实验管理方案；
9. 将联邦学习压缩为 future work，不再作为主项目内容；
10. 明确 soft BCE 与 Bernoulli KL 蒸馏的优化等价性，soft BCE 作为主公式；
11. 增加 distance-based uncertainty、low-variance/high-error 诊断和 overlap 区域经验判定规则；
12. 补充 Heckman correction、Bayesian reject inference、conformal prediction 等相关工作与 baseline；
13. 增加 propensity-matched accepted unlabeled control，验证 rejected 数据的独特价值；
14. 固定 LendingClub 年份切分方案，补充 rejected 数据代表性局限与实验并行调度策略；
15. 进一步明确 distillation 等价性、student 再校准、kNN-distance uncertainty 主方案、overlap 经验参数、int_rate 泄漏审计、IPW-only baseline 与关键点编号/参考文献一致性。
```

---

## 1. 执行摘要

传统信用评分研究通常默认训练集中的样本都有违约标签，即可以直接训练一个二分类模型预测 `default / non-default`。但真实信贷审批流程并不是这样：只有被批准放款的申请人才会产生后续还款表现，因此只有 **accepted applicants** 有可观测的违约标签；被拒贷的申请人没有获得贷款，自然也没有真实还款结果。因此，拒贷样本不是负样本，而是 **unlabeled samples under selection bias**。

本项目将信用评分问题重新定义为：

```text
Accepted applicants: labeled samples with observed repayment outcome
Rejected applicants: unlabeled samples with unobserved counterfactual repayment outcome
Goal: estimate calibrated PD for the full applicant population and optimize lending decisions
```

UCRI-CS 的核心思想是：

1. 不把拒贷样本简单当作坏客户或好客户；
2. 不只在已放款样本上训练 PD 模型；
3. 将拒贷推断建模为 selection-biased semi-supervised learning；
4. 使用 uncertainty-aware pseudo-labeling 控制伪标签污染；
5. 使用 calibrated distillation 将不确定性感知 teacher 模型的软预测蒸馏到轻量实用学生模型中；
6. 使用 decision-aware calibration 将 PD 预测与审批阈值、坏账率约束、预期利润和公平性指标统一评估；
7. 在真实拒贷数据和人造拒贷机制下分别评估，以避免“真实拒贷无标签导致无法验证”的审稿风险。

一句话定位：

> UCRI-CS is an uncertainty-calibrated semi-supervised reject inference framework that learns practical and decision-aware probability-of-default models from accepted labeled applicants and rejected unlabeled applicants under approval selection bias.

中文表述：

> UCRI-CS 将信贷拒贷推断建模为选择偏差下的半监督学习问题，通过不确定性感知伪标签、校准蒸馏和决策层校准，使模型不仅提升违约预测性能，而且能输出可用于审批决策的可靠违约概率。

---

## 2. 领域现状与投稿判断

### 2.1 为什么这个题目比“机器学习预测违约”更有论文价值

普通信用评分论文容易被审稿人认为是 benchmark engineering：换一个 LightGBM、XGBoost、FT-Transformer 或 GNN，然后在公开数据上报告 AUROC 提升。这类工作的问题是科学问题不够尖锐。

UCRI-CS 抓住了信贷建模中更本质的问题：

```text
训练标签只来自被批准样本；
模型应用对象却是全部申请人；
训练分布和应用分布不一致；
拒贷样本没有真实违约标签；
PD 预测需要校准后才能服务审批和定价；
审批策略还要考虑利润、坏账约束和公平性。
```

因此，本项目不是“预测违约”本身，而是解决：

> How can we estimate calibrated probability of default for the full applicant population when repayment labels are observed only for previously accepted applicants?

这比单纯追求 AUROC 更像一篇 CCF B/B+ 论文。

### 2.2 与已有 reject inference 方法的区别

传统 reject inference 常见路线包括：

| 方法 | 基本思想 | 局限 |
|---|---|---|
| Hard cut-off / augmentation | 给拒贷样本分配硬标签后加入训练 | 容易引入错误标签 |
| Parceling | 按风险分箱给拒贷样本分配坏账率 | 依赖人工分箱和强假设 |
| Fuzzy augmentation | 给拒贷样本分配软标签 | 软标签来源可能未校准 |
| Extrapolation | 用已放款样本外推到拒贷区域 | 高风险区域外推不稳定 |
| Semi-supervised self-training | 用模型生成伪标签迭代训练 | 错误高置信伪标签会污染模型 |
| Selection model / IPW | 建模被批准概率并重加权 | 高维非线性场景下权重方差大 |
| Heckman two-step correction | 先建模选择方程，再在结果方程中加入 inverse Mills ratio | 经典计量选择偏差校正，但要求较强的函数形式假设和合适的排除变量 |
| Bayesian reject inference | 用先验分布或贝叶斯层次模型表示拒贷样本的不确定性 | 先验设定敏感，公开大规模表格数据上复现和可扩展性需额外验证 |
| PU learning | 将部分有标签样本与无标签样本共同建模 | 需要明确 positive / negative / unlabeled 的语义，与 reject inference 的 accepted-good / accepted-bad / rejected 结构存在差异 |
| Conformal prediction / risk control | 给出分布无关的不确定性集合或风险控制 | 更适合作为风险控制和 abstention 补充，不能单独解决 rejected label 不可观测问题 |

UCRI-CS 的差异化不在于“首次做 reject inference”，而在于：

1. 把拒贷推断明确建模为 **selection-biased semi-supervised PD estimation**；
2. 用 **uncertainty-aware pseudo-labeling** 过滤或降权不可信拒贷伪标签；
3. 用 **calibrated distillation** 保证学生模型学习的是校准后的软违约概率，而非未校准 hard pseudo-label；
4. 用 **decision-aware evaluation** 检验模型是否真正改善审批策略，而不是只改善 AUROC。

## 3. 核心科学问题

### 3.1 已放款样本有标签，拒贷样本无标签

设申请人特征为 \(x\)，审批结果为 \(a \in \{0,1\}\)，其中：

```text
a = 1: accepted / approved / funded
a = 0: rejected / declined
```

真实违约结果为 \(y \in \{0,1\}\)，其中：

```text
y = 1: bad / default / charged-off / serious delinquency
y = 0: good / fully paid / non-default
```

现实中只有当 \(a=1\) 时，\(y\) 才能被观察到：

```text
Observed:
  accepted applicant: (x, a=1, y observed)

Unobserved:
  rejected applicant: (x, a=0, y missing)
```

因此训练数据不是普通 supervised learning，而是：

```text
Labeled set L = {(x_i, y_i): a_i = 1}
Unlabeled set U = {x_j: a_j = 0}
Selection indicator A = accepted / rejected
```

目标是在全体申请人分布上学习：

\[
p(y=1 \mid x)
\]

而不仅仅是在已放款分布上学习：

\[
p(y=1 \mid x, a=1)
\]

### 3.2 拒贷样本缺失机制不是随机缺失

拒贷样本没有违约标签不是 MCAR（missing completely at random）。审批策略本身依赖申请人风险特征，因此缺失机制更接近：

\[
P(a=1 \mid x, y) \neq P(a=1)
\]

这意味着 accepted-only 模型存在选择偏差：

```text
training distribution: accepted applicants
target distribution: all applicants
```

如果直接用 accepted-only 数据训练模型，模型可能在审批边界附近或被历史策略拒绝的人群上严重失真。

### 3.3 可识别性假设与适用边界

拒贷推断的核心困难是：真实目标量

\[
p(y=1\mid x)
\]

在只有 accepted applicants 拥有还款标签的条件下并非自动可识别。UCRI-CS 不应声称可以在没有额外假设的情况下恢复全体申请人的真实违约概率。本文采用如下清晰假设体系。

#### Assumption 1: Positivity / overlap

存在常数 \(\epsilon>0\)，使得在研究关注的特征支持域 \(\mathcal{X}_{overlap}\) 内：

\[
P(a=1\mid x)>\epsilon,\quad \forall x\in\mathcal{X}_{overlap}
\]

含义是：每类申请人至少有非零概率被历史审批策略批准。如果某些申请人群体从未被批准，历史数据无法提供其还款表现的经验支撑，模型只能把这些区域标记为 out-of-support / high uncertainty，而不能给出强可靠 PD。

**经验 overlap 判定规则：** 为避免 overlap 只是口头限定，实验中采用可操作定义。先训练审批倾向模型 \(e_\phi(x)=P(A=1\mid x)\)，并在 validation set 上做 calibration。样本 \(x\) 被标记为 in-overlap 当且仅当同时满足：

```text
1. calibrated propensity e_phi(x) ∈ [epsilon_low, 1 - epsilon_high]
   默认 epsilon_low = 0.05, epsilon_high = 0.05；
2. x 到 accepted training samples 的 kNN 距离不超过 accepted validation 距离分布的 95% 分位数；默认使用 k=10，并在 supplementary 中报告 k∈{5,10,20} 的敏感性；
3. 连续特征落在 accepted training 的 [1%, 99%] winsorized range 内，类别特征属于训练期已见类别。
```

主结果必须同时报告：

```text
in-overlap metrics
full-set metrics
out-of-support rate
in-overlap coverage among real rejected samples
```

如果 UCRI-CS 只在 full-set 上提升而 in-overlap 上不提升，不应声称 reject inference 有效；如果 out-of-support 样本比例很高，应将真实拒贷结论降级为分布诊断而非 PD 恢复。

#### Assumption 2: Conditional ignorability within observable policy features

在共同可观测申请特征 \(x\) 和可重构审批政策特征 \(s(x)\) 给定后，审批选择对潜在违约结果的额外依赖被弱化：

\[
Y(1) \perp A \mid x, s(x),\quad x\in\mathcal{X}_{overlap}
\]

其中 \(Y(1)\) 表示“如果被放款后”的潜在违约结果。该假设不是说真实审批完全随机，而是说审批策略中与违约相关的主要信息已被共同字段、风险评分、DTI、金额、就业年限、州/地区、时间等可观测变量捕获。

#### Assumption 3: Bounded extrapolation

对不满足 overlap 的样本，UCRI-CS 不做强外推，而是输出：

```text
high uncertainty
out-of-support flag
manual review recommendation
```

论文中所有关于 rejected applicants 的强结论必须限定在 \(\mathcal{X}_{overlap}\) 内。对 overlap 外区域，仅报告预测分布、覆盖率和不确定性，不声称已恢复真实 PD。

#### 适用边界声明

因此，UCRI-CS 的理论定位不是“无条件恢复所有拒贷人的真实违约标签”，而是：

```text
在历史审批政策有覆盖、共同特征可解释选择机制、并且模型不确定性可检测外推区域的条件下，
利用拒贷无标签样本改善审批边界附近的 PD 校准和决策效用。
```

该声明应放入论文 Introduction 和 Limitations，避免审稿人将项目理解为无假设 reject inference。

#### Conditional ignorability 的灵敏度分析

Assumption 2 是本项目最容易被审稿人攻击的理论假设，因为真实审批中可能存在无法在公开数据中观察到的软信息，例如人工审核意见、补充材料、地区信贷经理判断、借款人沟通质量、行业周期预判等。如果这些不可观测因素同时影响审批结果和未来违约，则

\[
Y(1) \not\perp A \mid x, s(x)
\]

UCRI-CS 不能保证恢复真实的全体申请人 PD。因此，本项目必须增加 **confounded rejection simulation**，人为构造不可观测选择因子 \(z\)：

\[
z_i = \rho\cdot y_i + \sqrt{1-\rho^2}\cdot \epsilon_i,\quad \epsilon_i\sim \mathcal{N}(0,1)
\]

并让模拟审批策略同时依赖可观测特征和不可观测因子：

\[
P(A_i=1\mid x_i,z_i)=\sigma(g(x_i)-\gamma z_i)
\]

其中 \(\rho\) 控制 hidden confounder 与违约标签的相关性，\(\gamma\) 控制审批策略对不可观测因子的依赖强度。这里 \(g(x)\) 必须预先固定并可复现，推荐两种设置：

```text
Primary g(x): 使用仅基于 shared-feature view 的 Logistic Regression 审批倾向模型 logit(e_phi(x))；
Robustness g(x): 使用规则化的 LightGBM propensity model logit(e_phi(x))，但不得与 student 主模型共享调参结果。

为避免 logit 在 \(e_\phi(x)\) 接近 0 或 1 时数值发散，所有 confounded rejection simulation 中均先将 calibrated propensity score clip 到 \([0.01,0.99]\)，再计算：

\[
g(x)=\log\frac{\mathrm{clip}(e_\phi(x),0.01,0.99)}{1-\mathrm{clip}(e_\phi(x),0.01,0.99)}
\]
```

实验设置：

```text
rho ∈ {0.0, 0.2, 0.4, 0.6}
gamma ∈ {0.0, 0.5, 1.0, 2.0}
rejection rate ∈ {20%, 40%, 60%}
```

报告指标：

```text
hidden-reject AUROC / PR-AUC / Brier / ECE
pseudo-label precision-coverage curve
performance degradation vs. gamma/rho
out-of-support rate and uncertainty response
```

论文中的解释原则：

```text
当 gamma=0 或 rho=0 时，选择机制近似可由可观测特征解释；
当 gamma 和 rho 增大时，conditional ignorability 被逐步破坏；
若 UCRI-CS 性能退化但 uncertainty 上升，说明模型能识别外推风险；
若性能退化且 uncertainty 不上升，则必须在 limitation 中承认模型无法检测该类隐藏选择偏差。
```

该实验不用于证明模型能解决不可观测混杂，而是用于量化理论假设失效时的退化边界。

### 3.4 伪标签可能污染模型

拒贷推断常用做法是先训练一个模型，再给 rejected applicants 打伪标签。但如果 teacher 模型在拒贷区域本身不可靠，高置信错误标签会导致 self-training 进入确认偏差循环：

```text
biased teacher -> wrong pseudo labels -> polluted student -> stronger bias
```

因此本项目的关键不是“给拒贷样本打标签”，而是：

> 只让模型学习那些不确定性低、校准后可信、且对决策边界有价值的拒贷信息。

### 3.5 AUROC 不是风控决策的充分指标

信贷审批需要的是可用 PD，而不是单纯排序分数。两个模型可能 AUROC 接近，但一个模型的概率严重过度自信，另一个模型校准良好。在定价、额度、坏账率约束和资本计量中，后者更有价值。

因此本项目把评价目标从：

```text
maximize AUROC
```

扩展为：

```text
maximize discrimination
+ improve probability calibration
+ optimize approval decision
+ control bad rate
+ improve expected profit
+ monitor subgroup/fairness behavior
```

---

## 4. 项目总体目标

UCRI-CS 的总目标是构建一个面向公开信贷数据的 **selection-bias-aware、semi-supervised、uncertainty-calibrated、decision-aware** 信用评分框架。

具体目标如下：

1. **拒贷推断建模：** 将 rejected applicants 建模为无标签样本，而不是负样本或简单丢弃样本。
2. **选择偏差校正：** 学习审批倾向模型 \(e(x)=P(a=1|x)\)，并通过 propensity weighting / representation balancing / policy simulation 缓解 accepted-only bias。
3. **不确定性感知伪标签：** 使用 ensemble variance、predictive entropy、margin uncertainty、conformal uncertainty 等指标筛选或加权拒贷伪标签。
4. **校准蒸馏：** 使用经过温度缩放、isotonic 或 beta calibration 后的 teacher soft labels 训练轻量实用学生模型。
5. **决策层优化：** 将 PD 预测转化为审批策略，评估 expected profit、bad-rate constraint、approval rate、KS、Brier、ECE 和 fairness。
6. **公开数据可复现：** 基于 LendingClub accepted/rejected 数据构造主实验，并用 Home Credit / HMDA / Fannie Mae / Freddie Mac 等数据做扩展验证。
7. **实用轻量模型输出：** 最终输出 PD、置信区间、不确定性分解、审批建议和决策解释；除非实际完成延迟与模型大小测试，正文不使用强工程承诺式的“deployable”表述。

---

## 5. 总体框架

```text
Data layer
  LendingClub accepted loans
  LendingClub rejected applications
  Optional: Home Credit / HMDA / Fannie Mae / Freddie Mac
        ↓
Data harmonization
  feature alignment
  application date alignment
  accepted/rejected indicator construction
  default label construction for accepted loans
  leakage-safe feature filtering
        ↓
Selection-bias modeling
  approval propensity model e(x)
  policy score reconstruction
  simulated rejection mechanism
        ↓
Teacher model layer
  calibrated LightGBM / CatBoost / FT-Transformer
  deep ensemble / bootstrap ensemble
  uncertainty estimation
        ↓
Semi-supervised reject inference
  uncertainty-aware pseudo-labeling
  soft-label weighting
  accepted + rejected training
        ↓
Calibrated distillation
  teacher probability calibration
  student model distillation
  calibration-aware loss
        ↓
Decision layer
  PD threshold optimization
  expected profit maximization
  bad-rate constraint
  fairness/subgroup audit
        ↓
Output
  calibrated PD
  uncertainty score
  approval/reject/review decision
  expected profit contribution
  reason-code style explanation
```

建议方法命名：

```text
UCRI-CS
= Uncertainty-Calibrated Reject Inference for Credit Scoring
```

或更论文式：

```text
UCRI
= Uncertainty-Calibrated Reject Inference
```

---

## 6. 数据集与数据标准化

## 6.1 主数据集：LendingClub accepted/rejected


### 6.1.0 主数据源、版本与复现口径

主实验建议固定一个 LendingClub 数据源，不混用 Kaggle 与 Figshare 等不同镜像。推荐写法：

```text
Primary source: Kaggle LendingClub accepted_2007_to_2018Q4.csv.gz and rejected_2007_to_2018Q4.csv.gz
Raw file checksum: report SHA256 in supplementary material
Snapshot date: record download date
Secondary source: Figshare mirror, only for robustness check if fields match
```

若 Kaggle 与 Figshare 字段、时间范围或清洗规则不一致，主文只报告主数据源结果，另一个数据源仅作为 external robustness。所有字段字典、删除样本比例、标签定义和时间切分必须在 supplementary 中固定，避免复现实验时出现“同名数据集但版本不同”的问题。

LendingClub 数据的优势是同时包含已通过贷款和被拒贷款记录。accepted loan records 可以构造还款表现标签，rejected application records 可作为无标签拒贷样本。项目主线应围绕该数据展开。

### 6.1.1 Accepted loan data

accepted loans 中通常包含：

```text
loan_amnt
term
int_rate
installment
grade / sub_grade
emp_title / emp_length
home_ownership
annual_inc
verification_status
issue_d
purpose
addr_state / zip_code
dti
delinq_2yrs
earliest_cr_line
fico_range_low / fico_range_high
open_acc
revol_bal
revol_util
total_acc
loan_status
```

其中 `loan_status` 可用于构造违约标签。

建议标签定义：

```text
bad = 1:
  Charged Off
  Default
  Late (31-120 days)
  Does not meet credit policy. Status: Charged Off

good = 0:
  Fully Paid
  Does not meet credit policy. Status: Fully Paid

excluded:
  Current
  In Grace Period
  Late (16-30 days)
  Issued
  loan_status unclear or not matured
```

是否纳入 `Late (16-30 days)` 可作为敏感性分析。



#### 标签定义与成熟样本敏感性分析

默认标签仅使用已经成熟、还款表现明确的 accepted loans：

```text
Bad/default: Charged Off, Default, Late (31-120 days) 等明确坏账状态
Good/non-default: Fully Paid
Excluded by default: Current, In Grace Period, Late (16-30 days), Issued, Does not meet credit policy 等状态不明确或未成熟样本
```

必须报告：

```text
每类 loan_status 的数量与比例；
被排除样本占 accepted 总量的比例；
被排除样本与保留样本在 loan_amnt、int_rate、grade、DTI、FICO/risk proxy、issue_d 上的分布差异；
```

标签敏感性至少包含三组：

| 设置 | Bad 定义 | Good 定义 | 用途 |
|---|---|---|---|
| Strict-matured | Charged Off / Default / Late 31-120 | Fully Paid | 主实验，标签最干净 |
| Early-delinquency-as-bad | 加入 Late 16-30 | Fully Paid | 检验轻度逾期归类影响 |
| Current-excluded robustness | 排除 Current | Fully Paid only | 默认设置 |
| Current-as-censored | 将 Current 作为 censored，不进入监督损失 | 检查未成熟样本偏差 |
| Random-current stress test | 将 Current 按相邻时间窗坏账率随机归入 good/bad，多 seed | 仅作敏感性压力测试 |

主结论必须在 Strict-matured 和 Early-delinquency-as-bad 两个定义下方向一致；若差异大，应解释为标签定义敏感而非方法稳定改进。

### 6.1.2 Rejected application data

rejected applications 通常包含较少字段：

```text
Amount Requested
Application Date
Loan Title
Risk_Score
Debt-To-Income Ratio
Zip Code
State
Employment Length
Policy Code
```

注意：accepted 和 rejected 的字段并不完全一致。为了避免模型只在 accepted-only 高维特征上表现好、却无法用于 rejected samples，需要构建两套特征视图。

| 特征视图 | 用途 | 说明 |
|---|---|---|
| Shared-feature view | 主拒贷推断实验 | 只使用 accepted/rejected 共同字段 |
| Accepted-rich view | accepted-only PD baseline | 使用 accepted loans 中更丰富特征 |
| Hybrid view | 半监督 teacher 辅助 | accepted-rich teacher 只用于 accepted 内部，不直接给 rejected 打标签 |
| Policy view | 审批倾向模型 | 使用共同字段预测 accepted/rejected |

主论文结论必须基于 shared-feature view，否则拒贷推断场景不成立。

Rejected 数据不包含真实还款表现，因此在主任务中只能作为 unlabeled samples 使用。

**Rejected 数据代表性边界：** LendingClub rejected 文件不应被理解为“所有被拒申请人”的完整集合。平台可能存在前置筛选、申请流程中途流失、看到利率或条款后主动放弃等未记录样本。因此，本文中的 rejected applicants 更准确地表示“进入公开 rejected 记录的未放款申请”，而不是整个潜在信贷需求人群。该局限需要在数据描述和 limitations 中同时声明。

#### Risk_Score 循环依赖审计

Rejected application data 中的 `Risk_Score` 是一个高风险字段。它可能是 LendingClub 或第三方信用评分，也可能是历史审批策略强依赖的输入或中间产物。若不加审计地使用，会造成循环依赖：

```text
Risk_Score 影响历史审批；
模型用 Risk_Score 预测审批或伪标签；
最终模型可能只是复制既有评分卡，而非学习违约风险。
```

因此，Risk_Score 必须设置三种实验口径：

| 设置 | Risk_Score 用法 | 目的 |
|---|---|---|
| No-RiskScore | 完全排除 | 主结论优先使用，避免循环依赖 |
| Input-RiskScore | 作为普通输入特征 | 检查其性能贡献 |
| Anchor-RiskScore | 不进入主模型，仅作为校准/分箱参考 | 检查是否可作为外部评分锚点 |

必须增加 baseline：

```text
Risk_Score-only Logistic Regression
Risk_Score-only binning / scorecard
Risk_Score + DTI simple model
```

如果 Risk_Score-only 已接近 UCRI-CS，则论文必须把贡献改写为“校准和决策层改进”，不能声称学到了显著新信号。

### 6.1.3 类别不平衡处理策略

信用违约是天然不平衡任务，bad rate 通常远低于 good rate。UCRI-CS 不应只报告 AUROC，因为 AUROC 对类别比例不敏感，可能掩盖少数类识别问题。训练和评估采用以下固定策略：

```text
Primary training: class-weighted BCE / logloss, positive weight = N_good / N_bad, capped at 20
GBDT baselines: scale_pos_weight 或 class_weight，并在 validation PR-AUC/Brier 上选择
Neural baselines: class-weighted BCE；focal loss 作为补充 baseline，不作为主模型默认
Sampling: 不对 test set 做任何重采样；SMOTE 只作为 baseline，不进入主模型
Threshold: classification threshold 不固定为 0.5，而在 validation set 上按 KS、expected profit 或 bad-rate constraint 分别选择
```

权重 cap=20 是防止极低 default rate 场景下梯度被少数 bad 样本主导的经验性选择。Supplementary 必须报告 cap ∈ {5, 10, 20, 50} 的敏感性；如果 bad rate 约 2% 导致原始权重接近 50，应明确说明 cap 会压低正类权重，主结论需对 cap 稳健。

必须报告：

```text
训练集、验证集、测试集 default rate；
PR-AUC baseline，即 positive class prevalence；
score decile bad rate；
不同 threshold selection rule 下的结果。
```

这样可以避免模型在总体 AUROC 上提升但对违约少数类没有实际收益。

## 6.2 辅助数据集

### 6.2.1 Home Credit Default Risk / Home Credit 2024 Model Stability

Home Credit 数据适合做传统 PD 建模、时间切片稳定性、校准漂移和多表特征工程，但它不包含真实 rejected applicants，因此不能作为拒贷推断主数据集。建议作为辅助实验：

```text
用途 1：验证 calibration-aware distillation 在 accepted-only PD 任务上是否稳定；
用途 2：验证时间外推、worst-period AUC、calibration drift；
用途 3：作为模型泛化性的 supplementary experiment。
```

### 6.2.2 HMDA mortgage application data

HMDA 数据包含贷款申请是否 originated / denied / withdrawn 等审批结果，还包含部分申请人与贷款属性。它适合用于：

```text
approval selection modeling
fairness / subgroup audit
approval policy simulation
```

但 HMDA 通常不直接包含后续违约表现，因此不适合作为主 PD 标签数据。可用于补充“审批决策公平性”分析。

### 6.2.3 Fannie Mae / Freddie Mac loan performance data

Fannie Mae 与 Freddie Mac 公开贷款表现数据适合做 mortgage default / delinquency prediction 和 out-of-time PD calibration，但通常缺少完整拒贷申请样本。可作为：

```text
external PD calibration benchmark
time stability benchmark
mortgage-specific robustness benchmark
```

## 6.3 统一样本 schema

建议把所有样本统一成如下 schema：

```text
application_id
source_dataset
application_date
issue_date
loan_amount
loan_purpose
employment_length
dti
state
zip3
risk_score
policy_code

accepted_indicator
default_label
default_observed_indicator
loan_status_raw
maturity_indicator

train_valid_test_split
time_period
simulated_reject_indicator
propensity_score
pseudo_label
pseudo_label_confidence
pseudo_label_uncertainty
calibrated_pd
decision_label
```

其中：

```text
accepted_indicator = 1 表示已批准/已放款
default_observed_indicator = 1 表示真实还款表现可观察
```

对于真实 rejected samples：

```text
accepted_indicator = 0
default_observed_indicator = 0
default_label = missing
```

## 6.4 数据泄漏控制原则

信贷数据非常容易发生时间泄漏和结果泄漏。必须遵守：

1. **禁止使用贷后变量：** 如 recoveries、collection_recovery_fee、last_pymnt_d、total_pymnt 等。
2. **禁止使用结果派生字段：** 任何在放款后才知道的字段不得用于训练。
3. **按时间切分：** 主实验使用 application_date / issue_d 做 out-of-time split。
4. **拒贷推断只用共同字段：** 不能用 rejected 数据中不存在的 accepted-only 字段给拒贷样本打标签。
5. **校准集隔离：** calibration set 不参与模型训练。
6. **伪标签隔离：** 生成 pseudo-label 的 teacher 不能在同一批样本上直接训练并生成伪标签，应采用 out-of-fold 机制。
7. **审批倾向模型隔离：** propensity model 只预测 accepted/rejected，不得使用违约标签或贷后变量。

---



### 6.5 禁止特征清单与贷后泄漏审计

LendingClub accepted 数据中存在大量贷后字段，这些字段在申请审批时不可获得，必须从所有模型、baseline、policy simulation 和特征工程中排除。

| 禁止字段类别 | 示例字段 | 禁止原因 |
|---|---|---|
| 还款过程字段 | `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` | 贷后还款结果，直接泄漏标签 |
| 回收催收字段 | `recoveries`, `collection_recovery_fee` | 坏账后催收结果，强标签泄漏 |
| 最后还款/信用更新字段 | `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`, `last_credit_pull_d` | 发生在审批之后 |
| 结清/账户状态字段 | `loan_status`, `hardship_flag`, `debt_settlement_flag`, `settlement_status` | 标签或贷后处置代理 |
| 贷后余额字段 | `out_prncp`, `out_prncp_inv` | 贷款发放后的余额信息 |
| 逾期历史更新字段 | `acc_now_delinq`, `delinq_amnt` 若时间戳晚于申请 | 需要确认时间口径，否则可能泄漏 |
| 定价/审批后可能生成字段 | `int_rate`, `installment`, `grade`, `sub_grade` | 若字段在最终授信定价后才产生，或 rejected 数据中不可用，则不得进入 shared-feature view；可仅用于 accepted-rich baseline，并必须做含/不含敏感性分析 |

允许特征必须满足：

```text
1. 在申请时可获得；
2. accepted 与 rejected 中均可映射，或仅用于 accepted-only baseline；
3. 不由未来还款行为、催收行为、贷后状态衍生；
4. 在特征字典中明确记录可用时间点。
```

所有实验脚本应包含 leakage audit：若 forbidden feature 出现在训练矩阵中，程序直接报错并停止。
**关于 `int_rate` 的特殊处理：** LendingClub accepted 数据中的 `int_rate` 可能是风险定价后的结果，而 rejected 申请通常没有最终利率。如果无法确认该字段在审批前即可确定，则主 shared-feature view 中排除 `int_rate`、`installment`、`grade/sub_grade`；它们只允许进入 accepted-rich baseline，并必须报告含/不含这些字段的敏感性结果。

## 7. 模型设计

## 7.1 问题定义

给定申请人特征 \(x_i\)、审批结果 \(a_i\)、违约标签 \(y_i\)，训练集中只有 \(a_i=1\) 的样本观测到 \(y_i\)。

定义：

\[
L = \{(x_i,y_i): a_i=1, y_i \text{ observed}\}
\]

\[
U = \{x_j: a_j=0, y_j \text{ unobserved}\}
\]

目标学习：

\[
f_\theta(x) = P(y=1 \mid x)
\]

并同时输出：

```text
pd_score: 原始违约概率
calibrated_pd: 校准后违约概率
uncertainty: 模型不确定性
decision: approve / reject / manual review
explanation: 决策解释或 reason codes
```

## 7.2 Selection model：审批倾向建模

首先训练一个审批倾向模型：

\[
e_\phi(x)=P(a=1 \mid x)
\]

该模型用于估计申请人进入 accepted labeled set 的概率。它不直接预测违约，而是预测历史审批策略。

可选用途：

1. **Inverse propensity weighting：**

\[
w_i = \frac{1}{\max(e_\phi(x_i), \epsilon)}
\]

用于降低 accepted-only 样本选择偏差。

2. **Representation balancing：**  
   使 accepted 与 rejected 的表征分布更加接近。

3. **Policy simulation：**  
   根据 \(e_\phi(x)\) 构造人造拒贷机制，用于可验证实验。

审批倾向模型建议使用：

```text
Logistic Regression
LightGBM
CatBoost
FT-Transformer
```

## 7.3 Teacher model：不确定性感知 PD teacher

teacher 模型在 accepted labeled samples 上训练：

\[
t_m(x) = P_m(y=1 \mid x, a=1)
\]

其中 \(m=1,\dots,M\) 表示 ensemble 中第 \(m\) 个模型。

建议 teacher 组合：

```text
LightGBM teacher
CatBoost teacher
FT-Transformer teacher
MLP teacher
bootstrap ensemble or cross-fold ensemble
```

输出 ensemble mean：

\[
\bar{p}(x)=\frac{1}{M}\sum_{m=1}^{M} t_m(x)
\]

模型不确定性：

\[
u_{var}(x)=\frac{1}{M}\sum_{m=1}^{M}(t_m(x)-\bar{p}(x))^2
\]

预测熵：

\[
u_{ent}(x)=-\bar{p}(x)\log \bar{p}(x)-(1-\bar{p}(x))\log(1-\bar{p}(x))
\]

margin uncertainty：

\[
u_{margin}(x)=1-|2\bar{p}(x)-1|
\]

仅依赖 ensemble variance 存在局限：如果所有 teacher 成员属于相近模型族，它们可能在 OOD / rejected 区域同时犯错但方差仍然很低。因此增加 distance-based uncertainty 作为第四类外推风险信号：

\[
u_{dist}(x)=\min_{x_i\in A_{train}} d_\Sigma(x,x_i)
\]

其中 \(d_\Sigma\) 的主方案采用 **standardized-feature kNN distance**，而不是 Mahalanobis distance。理由是 LendingClub 共同字段包含混合连续/类别变量，Mahalanobis 需要稳定估计协方差矩阵，在高维 one-hot 与 accepted/rejected 协方差不一致时容易不稳。主实现固定为：

```text
连续特征：robust z-score 标准化；
类别特征：one-hot 或 target-encoding 后统一标准化；
kNN distance：到 accepted training set 最近 k=10 个样本的平均距离；
normalization：在 accepted validation distance 分布上做 quantile normalization。
```

Mahalanobis distance 仅作为 supplementary robustness，不作为主方法。

综合不确定性不直接线性相加原始量，因为 variance、entropy、margin 和 distance 的尺度不同。先在 validation set 上做分位数归一化：

\[
\hat{u}_k(x)=Q_k(u_k(x))\in[0,1]
\]

其中 \(Q_k\) 是第 \(k\) 个不确定性分量的 empirical CDF / quantile transform。再组合：

\[
u(x)=\alpha_1 \hat{u}_{var}(x)+\alpha_2 \hat{u}_{ent}(x)+\alpha_3 \hat{u}_{margin}(x)+\alpha_4 \hat{u}_{dist}(x)
\]

\(\alpha\) 不作为自由大网格搜索参数，优先采用两种策略：

```text
Default: alpha = (1/4, 1/4, 1/4, 1/4)
Learned: 在 simulated rejection validation 上最大化 pseudo-label precision-coverage AUC
```

消融必须报告：

```text
variance only
entropy only
margin only
distance only
equal-weight combination
learned-alpha combination
low-variance / high-error region diagnostic in simulated rejection
```

其中 low-variance / high-error diagnostic 在 hidden-label simulated rejection 中计算：把 teacher variance 位于最低 20% 但预测错误或 calibration residual 位于最高 20% 的样本定义为 “confidently wrong” 区域，报告其比例、特征分布和是否被 distance uncertainty 捕获。

### 7.3.1 Teacher 在 rejected 分布上的 covariate shift 风险

teacher 在 accepted labeled samples 上训练，因此其原始输出更准确地表示：

\[
P(Y=1\mid X=x,A=1)
\]

而不是无条件全体申请人目标：

\[
P(Y=1\mid X=x)
\]

当 rejected applicants 与 accepted applicants 的特征分布差异较大时，teacher 对 rejected 的伪标签可能系统性偏误。UCRI-CS 不应默认 teacher 在 rejected 上可靠，而应通过四类机制控制风险：

1. **Overlap filtering：** 只对 propensity score 位于训练支持域内的 rejected 样本产生高权重伪标签；
2. **Uncertainty down-weighting：** ensemble disagreement、entropy、margin uncertainty 和 distance uncertainty 高的 rejected 样本降低权重；
3. **Cross-population calibration check：** 在 simulated rejection 中，把 hidden rejected-like 样本作为 proxy，检查 teacher soft label 的 ECE、Brier 和 calibration slope；
4. **Confidently-wrong diagnostics：** 检查低 variance 但高 error 的样本是否被 distance / overlap 机制识别。

需要报告：

```text
MMD/PSI: accepted vs real rejected feature distribution
teacher uncertainty: accepted validation vs real rejected
cross-population ECE: accepted-like vs hidden-rejected-like
pseudo-label coverage: overlap 内/外分别统计
low-variance/high-error region size under simulated rejection
```

论文表述必须区分：

```text
Within-accepted performance: teacher 在 accepted validation/test 上的判别与校准；
Rejected-region reliability: 只能通过 overlap、uncertainty 与 simulated hidden-label 间接支持；
Out-of-support rejected: 不参与强伪标签训练，只输出 high uncertainty / manual review flag。
```

## 7.4 Calibration module：teacher 概率校准

teacher 在 accepted 上的输出可能过度自信，因此先在 accepted validation set 上做 within-accepted calibration。但必须强调：即使 within-accepted calibration 完美，teacher 对 rejected 样本的预测仍可能存在由 covariate shift 和 selection bias 导致的系统偏误。这部分只能通过 overlap filtering、distance/ensemble uncertainty down-weighting 和 simulated rejection 中的 cross-population calibration check 缓解，不能被常规 calibration 自动消除。

#### 主校准方案与信息流边界

主文预设 **temperature scaling** 为 primary calibration method，理由是：参数少、过拟合风险低、适合 logits 输出、便于跨模型复现。Isotonic regression、beta calibration 和 spline calibration 作为 supplementary robustness，不在主文中挑最佳方法报告。

校准分两层：

| 层级 | 目标 | 验证位置 | 可支持的结论 |
|---|---|---|---|
| Within-accepted calibration | 校准 \(P(Y=1\mid X,A=1)\) | accepted validation/test | teacher/student 在 accepted 分布上概率可靠 |
| Cross-population calibration | 检查 rejected-like 区域的校准迁移 | simulated hidden-reject validation/test | 在特定模拟选择机制下，伪标签和 student 是否在 rejected-like 样本上保持校准 |

需要强调：within-accepted calibration 不能自动推出 real rejected calibration。真实 rejected 无标签，因此真实拒贷区域的 PD 校准性无法被直接验证，只能通过 overlap diagnostics、uncertainty 和 simulated rejection 间接评估。

可用方法：

```text
Primary: Temperature scaling
Robustness: Platt scaling / Isotonic regression / Beta calibration / Spline calibration
Supplementary: Conformal calibration or conformal risk control for abstention analysis
```

对 teacher logits \(z(x)\) 做 temperature scaling：

\[
p_T(x)=\sigma\left(\frac{z(x)}{T}\right)
\]

其中 \(T\) 在 validation set 上通过最小化 NLL 或 Brier score 学习。

校准后 teacher 输出：

\[
\tilde{p}(x)=Calibrate(\bar{p}(x))
\]

用于 rejected unlabeled samples 的 pseudo-labeling 和 distillation。

## 7.5 Uncertainty-aware pseudo-labeling

对 rejected samples \(x_j \in U\)，teacher 生成软伪标签：

\[
\tilde{y}_j = \tilde{p}(x_j)
\]

但不是所有拒贷样本都进入训练。定义伪标签权重：

\[
q_j = \exp(-\gamma u(x_j)) \cdot \mathbb{I}[u(x_j)<\tau_u]
\]

其中：

```text
u(x_j): 综合不确定性
tau_u: 不确定性阈值
gamma: 温度系数
q_j: 伪标签训练权重
```

也可以采用三分决策：

| 条件 | 处理 |
|---|---|
| 低不确定性 + 高违约概率 | 作为 high-risk soft bad 样本 |
| 低不确定性 + 低违约概率 | 作为 high-confidence good 样本 |
| 高不确定性 | 不打硬标签，只用于一致性正则或人工复核分析 |

该设计的关键是：

```text
Pseudo-labels are soft, calibrated and uncertainty-weighted.
```



### 7.5.1 伪标签阈值 \(\tau_u\) 与 coverage-sensitive 训练

伪标签阈值 \(\tau_u\) 是关键超参数，不能只报告单点最优结果。默认设置：

```text
tau_u ∈ {0.1, 0.2, 0.3, 0.4, 0.5} after quantile-normalized uncertainty
pseudo-label coverage target ∈ {20%, 40%, 60%, 80%}
```

主文使用 validation simulated rejection 选定一个默认 \(\tau_u\)，并在所有真实拒贷实验中固定。Supplementary 报告：

```text
performance vs tau_u under each simulated rejection mechanism
whether the same tau_u generalizes across mechanisms
precision-coverage and calibration-coverage curves
```

若不同机制下最优 \(\tau_u\) 差异很大，应弱化“通用阈值”主张，改为报告 coverage-constrained 策略。若固定 \(\tau_u\) 导致 pseudo-label coverage < 10%，不得只报告该低覆盖率结果；应额外报告 coverage-constrained 策略，例如选择 uncertainty 最低的 30% rejected 样本，并明确其 precision-coverage / calibration-coverage trade-off。

## 7.6 Student model：轻量实用学生模型

student 模型是最终论文主评估模型。建议优先使用：

```text
LightGBM
CatBoost
Tabular MLP
FT-Transformer
```

如果目标是更贴近风控实践，主学生模型建议使用 LightGBM / CatBoost，因为：

```text
推理快；
特征重要性和 SHAP 解释成熟；
在表格数据上强；
容易审计。
```

**蒸馏必要性说明：** student 不是为了和单个 LightGBM teacher 做同义替换，而是为了解决三个问题：

```text
1. 推理成本：teacher ensemble 需要 M 次前向传播和多个模型文件；student 只需 1 次前向传播。
2. 分布适配：teacher 在 accepted 分布上训练，直接用于 rejected 区域会有 covariate shift；student 通过 uncertainty-weighted rejected soft labels、overlap filtering 和可选 representation balancing，使决策边界在全体申请人 shared-feature view 上重新正则化。
3. 模型交付：student 固定为单模型，可输出统一 reason codes、统一 calibration report 和统一 threshold policy；teacher ensemble 主要承担训练期 teacher / uncertainty estimator 角色。
```

论文中应报告工程量化指标：

| 模型 | 推理次数 | 模型大小估计 | 单万样本推理时间 | 备注 |
|---|---:|---:|---:|---|
| Teacher ensemble, M=5 | 5 次 | 约 50–250 MB（5 个 10–50 MB GBDT 模型） | 需实测；理论约为 student 的 5 倍前向开销 | 训练期 teacher 与不确定性估计器 |
| Student LightGBM/CatBoost | 1 次 | 约 10–50 MB（依树数、深度和特征数变化） | 需实测；主文报告 CPU batch inference | 主结果和决策模拟使用的 lightweight 模型 |

这里“可部署”不应写成强工程承诺，建议在论文中使用 **practical / lightweight / audit-friendly**，除非实际测量了延迟和模型大小。

student 的训练损失：

\[
L = L_{sup} + \lambda_1 L_{distill} + \lambda_2 L_{calib} + \lambda_3 L_{balance} + \lambda_4 L_{decision}
\]

#### λ 权重设定策略

为避免 4 个 λ 权重形成不可控网格搜索，采用“固定主干 + 少量敏感性分析”的策略：

| 权重 | 默认策略 | 推荐搜索范围 | 说明 |
|---|---|---|---|
| \(\lambda_1\) distillation | 由伪标签平均置信度自动缩放 | {0.1, 0.3, 1.0} | 伪标签越不确定，整体蒸馏权重越低 |
| \(\lambda_2\) calibration | validation Brier/ECE 选择 | {0.01, 0.05, 0.1} | 防止校准项压制排序能力 |
| \(\lambda_3\) balance | 默认 0，作为可选模块 | {0, 0.01, 0.05} | 只有 accepted/rejected 分布差异显著时启用 |
| \(\lambda_4\) decision | 不进入主训练，后处理阈值优化为主 | {0, 0.01} | 避免利润假设污染 PD 学习 |

更稳妥的主论文设置：

```text
Main model: L_sup + λ1 L_distill + λ2 L_calib
Optional analysis: + λ3 L_balance
Decision layer: post-hoc threshold optimization, not default differentiable training loss
```

补充实验报告 λ sensitivity heatmap，重点展示 \(\lambda_1\) 与 \(\lambda_2\) 在合理范围内性能稳定。若使用自动权重，可采用 uncertainty-weighted multi-task learning，但不作为唯一方案，避免增加方法复杂度。

### 监督损失

\[
L_{sup}=\sum_{i\in L} w_i \cdot BCE(y_i, f_\theta(x_i))
\]

### 蒸馏损失

主公式采用 weighted soft BCE：

\[
L_{distill}=-\sum_{j\in U} q_j\left[\tilde{y}_j \log f_\theta(x_j)+(1-\tilde{y}_j)\log(1-f_\theta(x_j))\right]
\]

其中 \(\tilde{y}_j=\tilde{p}(x_j)\) 是 calibrated teacher soft label，\(q_j\) 是 uncertainty/overlap 权重。也可从 Bernoulli KL 角度解释该损失：

\[
KL(\tilde{p}\|f_\theta)=\tilde{p}\log\frac{\tilde{p}}{f_\theta}+(1-\tilde{p})\log\frac{1-\tilde{p}}{1-f_\theta}
\]

其中 \(\tilde{p}\log\tilde{p}+(1-\tilde{p})\log(1-\tilde{p})\) 对 \(\theta\) 为常数。因此，在 Bernoulli 输出下，最小化 KL 与最小化 soft BCE 在优化 student 参数时等价。正文使用 soft BCE，KL 仅作为概念解释，避免误解为两个独立损失项。

### 校准损失

可使用 differentiable ECE surrogate 或 Brier loss：

\[
L_{calib}=\sum_i (f_\theta(x_i)-y_i)^2
\]

### 表征平衡损失

可选项，用于 accepted/rejected 表征分布差异较大时：

\[
L_{balance}=D(h(X_A), h(X_R))
\]

其中 \(D\) 可取 MMD 或 CORAL。该项默认不启用，即 \(\lambda_3=0\)。只有当 accepted/rejected 的 MMD/PSI 明显偏大，且消融证明有效时，才纳入完整模型。

### 决策损失

不建议主模型直接优化利润损失，因为利润参数依赖外生假设。更稳妥做法：PD 模型先训练和校准，再在 validation set 上做 post-hoc threshold optimization。若一定使用 differentiable decision loss，应作为 supplementary experiment。

#### 利润参数锚定

利润分析不作为唯一主结论，必须报告参数来源和敏感性区间。建议设置：

```text
LGD grid: 20%, 35%, 45%, 60%, 75%, 90%
funding_cost APR: 2%, 4%, 6%, 8%
servicing_cost: 0%, 1%, 2% of loan amount
prepayment / duration haircut: 50%, 75%, 100% expected term
```

其中 45% 可作为无抵押 senior exposure 的传统监管锚点，40%/45% 可作为 Basel 框架下不同 senior unsecured exposure 的参考值；P2P unsecured personal loan 的真实 LGD 可能更高，因此 60%/75%/90% 作为压力情景。

主文展示 profit frontier 的形状是否稳定，而不是只报告某一组 LGD 下的最优利润。

### Student 输出的交付前校准检查

Student 训练完成后，需要在 accepted validation/calibration set 上单独检查其 ECE、Brier score 和 reliability diagram。即使训练目标包含 \(L_{calib}\)，supervised label loss 与 distillation soft-label loss 的混合仍可能导致 student 输出重新失校准。若 student 的 ECE 高于 teacher 的 within-accepted ECE，或高于预设阈值，则对 student logits 额外应用一次 temperature scaling，温度 \(T_s\) 仅在 accepted validation/calibration set 上学习。该步骤是 student 输出交付前的标准后处理；同时报告 before/after calibration 结果，避免把后处理收益混入主训练模块贡献。

## 7.7 Decision-aware calibration

最终模型不是简单输出 PD，而是输出决策建议：

```text
approve
reject
manual review
```

定义三个决策阈值：

```text
theta_low: 低风险批准阈值
theta_high: 高风险拒绝阈值
tau_decision: 决策阶段的不确定性阈值
```

其中 \(\tau_{decision}\) 不等同于伪标签训练阈值 \(\tau_u\)。\(\tau_u\) 控制 rejected 样本是否进入蒸馏训练，通常应更保守；\(\tau_{decision}\) 控制线上或离线策略中的人工复核边界，可在 validation set 上根据 approval rate、bad rate、manual review rate 和 expected profit 联合选择。默认报告 \(\tau_{decision}\in\{\tau_u,\ 1.25\tau_u,\ 1.5\tau_u\}\) 的敏感性。

决策规则：

```text
if calibrated_pd <= theta_low and uncertainty <= tau_decision:
    approve
elif calibrated_pd >= theta_high:
    reject
else:
    manual review
```

该机制对应风控中的 reject option / human-in-the-loop 审批流程。

可优化目标：

```text
maximize expected profit
subject to:
  bad_rate <= target_bad_rate
  approval_rate >= minimum_approval_rate
  ECE <= calibration_threshold
  subgroup disparity <= fairness_threshold
```



#### Oracle profit baseline 与利润简化边界

利润函数不应只报告绝对值，因为 LGD、资金成本、运营成本、提前还款率和监管资本成本均依赖外部假设。为避免“参数恰好有利于本方法”的质疑，必须加入 oracle profit baseline：

```text
Oracle ranking: 使用真实 y 对样本做完美排序，给出理论利润上限
Random approval: 随机审批，给出下限
Historical policy proxy: 使用原始审批/grade/Risk_Score 近似历史策略
Model policies: 各方法按预测 PD 或 expected profit 排序
```

报告形式优先使用：

\[
Profit\ Ratio = \frac{Profit_{model}-Profit_{random}}{Profit_{oracle}-Profit_{random}}
\]

以及相对于 oracle 的 gap：

\[
Oracle\ Gap = 1 - Profit\ Ratio
\]

这样即使利润参数变化，仍能观察各方法距离理论上限的相对位置。

Limitations 中必须承认：当前利润函数是单期、单笔贷款近似，未显式建模客户生命周期价值、竞品流失、提前还款行为、监管资本占用和宏观资金成本变化。决策实验用于比较模型排序和校准的相对价值，不应被解释为真实银行利润预测。

## 7.8 解释模块

面向风控论文和业务落地，解释模块建议采用：

```text
SHAP values for LightGBM/CatBoost
permutation importance
monotonicity audit
reason-code style top features
counterfactual explanation
```

输出格式：

```text
application_id
calibrated_pd
decision
uncertainty
top_positive_risk_factors
top_negative_risk_factors
counterfactual_suggestion
```

示例：

```text
Applicant 123:
  calibrated_pd = 0.184
  uncertainty = 0.037
  decision = manual review
  risk drivers:
    high DTI
    short employment length
    requested amount above peer median
  protective factors:
    stable risk score
    low historical rejection propensity
```

需要注意：公开数据上的解释只作为模型可解释性演示，不能声称符合真实银行监管 reason code 要求。

---

## 8. 实验设计

## 8.1 实验总原则

实验要回答以下问题：

1. UCRI-CS 是否优于 accepted-only PD 模型？
2. 使用 rejected unlabeled samples 是否真的带来增益？
3. uncertainty-aware pseudo-labeling 是否减少伪标签污染？
4. 校准蒸馏是否改善 Brier/ECE，而不是只改善 AUROC？
5. 决策层是否提升 expected profit 和 approval-bad-rate trade-off？
6. 在人造拒贷机制下，模型能否恢复隐藏 rejected labels？
7. 方法是否在时间外推和分布漂移下稳定？

## 8.2 Protocol 1：Accepted-only out-of-time PD benchmark

目的：建立基础违约预测能力。

设置：

```text
Dataset: LendingClub accepted loans
Train: issue_d 2012-2014
Validation: issue_d 2015
Test-normal: issue_d 2016-2017
Test-extended: issue_d 2018-2019
Test-structural-break: issue_d 2020, reported separately as stress test
Note: this split must stay consistent with §8.5.1.
Labels: matured loan_status only
Features: accepted-rich view and shared-feature view separately
```

报告：

```text
AUROC
PR-AUC
KS
Brier score
ECE
calibration slope/intercept
bad rate by score decile
```

作用：

```text
证明模型在普通 PD 任务上不弱；
提供 accepted-only baseline；
区分 accepted-rich view 和 shared-feature view 的性能差距。
```

## 8.3 Protocol 2：Real rejected semi-supervised learning

目的：使用真实 rejected applicants 作为无标签样本参与训练。

设置：

```text
Labeled set: accepted loans with observed default labels
Unlabeled set: rejected applications
Feature view: shared-feature view
Teacher: accepted labeled training only
Pseudo-labeling: rejected samples only
Student: accepted labels + weighted rejected pseudo-labels
Test: future accepted loans
```

由于真实 rejected samples 没有违约标签，主评价不能直接在 rejected 上计算准确率，而是在 future accepted test 和 rejected 分布诊断上评估：

```text
discrimination improvement on future accepted
calibration improvement on future accepted
decision utility improvement on future accepted
score distribution shift for rejected
uncertainty distribution for rejected
approval policy simulation
overlap-region coverage
out-of-support rate
```

需要明确声明：Protocol 2 不能直接证明 rejected group 的真实 PD 更准，它只能证明引入 rejected unlabeled samples 后，模型在未来 accepted 样本、审批边界和分布诊断上是否更合理。真实拒贷效果的直接验证必须依赖 Protocol 3 的 hidden-label simulated rejection。

此外，应把 real rejected 的结论写成 diagnostics，而不是 accuracy claim：

```text
Real rejected 无标签，因此不能验证其真实 PD 校准性；
若模型给 rejected 更高或更低 PD，只能结合 overlap、uncertainty、Risk_Score monotonicity 和 simulated rejection 解释；
任何“发现误拒客户”的结论必须限定为候选发现，而不是确认标签。
```


真实 rejected 分析的推荐输出：

```text
1. rejected score distribution 是否相对 accepted 向高风险移动；
2. 低风险低不确定性 rejected 样本比例，即潜在误拒候选；
3. 高不确定性 rejected 样本比例，即不可识别/需人工复核区域；
4. 使用 No-RiskScore 与 Input-RiskScore 时 rejected PD 分布差异；
5. 与历史审批倾向 e(x) 的 monotonic consistency。
```

## 8.4 Protocol 3：Simulated rejection with hidden labels

这是最关键的可验证实验。

从 accepted loans 中构造人造拒贷样本：

```text
1. 先在历史 accepted loans 上训练 policy model e(x)。
2. 按风险分数、DTI、loan amount、employment length 等变量模拟审批策略。
3. 将一部分有真实 y 的样本标记为 simulated rejected。
4. 训练时隐藏这些样本的 y，只保留 x。
5. 测试时恢复 y，用于评估 reject inference 是否正确。
```

模拟机制必须与 UCRI-CS 的 teacher/student 算法族解耦，避免“用 LightGBM 模拟，再用 LightGBM 恢复”的循环验证。建议设置五类：

| 机制 | 模拟方式 | 与主模型关系 | 目的 |
|---|---|---|---|
| Linear-observed policy | Logistic regression on shared features | 与 LightGBM/CatBoost student 解耦 | 基础 MAR-like 场景 |
| Rule-based policy | DTI、金额、就业年限、州的人工规则 | 完全外生 | 检验规则拒贷恢复能力 |
| Score-band policy | 按 Risk_Score 分段拒贷 | 高度贴近真实评分卡 | 检验风险评分驱动的拒贷 |
| Geography/time policy | 按州/时间窗口截断或降采样 | 外生分布偏移 | 检验非风险变量导致的选择偏差 |
| Nonlinear black-box policy | Random Forest 或 shallow MLP | 不使用 student 同族模型 | 检验复杂非线性拒贷 |

禁止把主 student 的同族最优模型直接作为唯一模拟机制。若主 student 是 LightGBM，则 nonlinear policy 不应只用 LightGBM；可以使用 Logistic/RF/MLP 多机制组合。

还需设置 policy severity：

```text
rejection rate: 20%, 40%, 60%, 80%
overlap level: high / medium / low
policy noise: 0%, 10%, 20% random approval override
```

这样可以报告“模拟机制与模型假设差异变大时，UCRI-CS 性能如何退化”，防止审稿人认为方法只适配某一种人造拒贷规则。



#### Simulated rejection 的外推边界

必须明确：Protocol 3 评估的是 **在 accepted 群体内部，人为隐藏标签后方法能否恢复排序、校准和决策效用**，而不是直接证明真实 rejected 群体的绝对 PD 水平被恢复。原因是 accepted loans 已经经过历史审批策略筛选，即使从 accepted 中再模拟 rejection，其支持域仍可能比真实 rejected 更接近 approved population。

因此，每个 simulated rejection 机制都要同时报告：

```text
MMD / PSI between simulated rejected and real rejected
MMD / PSI between simulated rejected and accepted train
feature coverage overlap with real rejected
score distribution comparison under historical Risk_Score or grade proxy
```

若 simulated rejected 与 real rejected 差异很大，论文应将该机制解释为“可验证的内部压力测试”，而不是“真实拒贷验证”。

评价指标：

```text
hidden-reject AUROC
hidden-reject PR-AUC
hidden-reject Brier
hidden-reject ECE
pseudo-label precision
pseudo-label coverage
bad-rate estimation error
decision profit on hidden rejected group
```

这是回应审稿人质疑的核心：

> 真实拒贷无标签不可验证，所以我们用 controlled rejection simulation 评估方法是否能恢复被隐藏的还款结果。

## 8.5 Protocol 4：Policy shift and temporal stability

目的：检验不同时间段审批策略变化时模型稳定性。

设置采用 §8.5.1 的固定年份切分方案，以保证与全局 out-of-time protocol 一致：

```text
Train: accepted loans issued in 2012-2014
Validation: accepted loans issued in 2015
Test-normal: accepted loans issued in 2016-2017
Test-extended: accepted loans issued in 2018-2019
Test-structural-break: accepted loans issued in 2020, reported separately as stress test
Rejected unlabeled: aligned by application date to the corresponding train/validation/test periods
```

指标：

```text
period-wise AUROC
period-wise PR-AUC
worst-period AUROC
Brier drift
ECE drift
PSI of score distribution
approval rate drift
bad rate by time period
```

该实验可吸收你原先 StableGraphPFN 的时间稳定性思想。



### 8.5.1 正常时间漂移与结构性断点

LendingClub 跨年份数据包含平台策略变化、投资者结构变化、宏观利率环境变化和 2020 年疫情冲击等潜在结构性断点。时间外推实验应区分两类测试：

| 类型 | 示例切分 | 解释 |
|---|---|---|
| Normal drift | train 2012-2014, val 2015, test 2016-2017 | 用于判断模型在常规时间漂移下的稳定性 |
| Extended drift | train 2012-2014, val 2015, test 2018-2019 | 检验较长时间跨度的退化 |
| Structural break | test 2020 | 单独报告，不作为模型优劣的主要判据 |

主文结论优先基于 normal/extended drift。结构性断点结果应单独标注为 stress test，避免把宏观制度变化误解释为模型缺陷。

## 8.6 Protocol 5：Decision-aware approval simulation

目的：证明模型能改善实际审批策略。

给定一组审批阈值或优化约束：

```text
target_bad_rate ∈ {5%, 8%, 10%, 12%}
approval_rate constraint ∈ {20%, 30%, 40%, 50%}
LGD ∈ {20%, 35%, 45%, 60%, 75%, 90%}
```

比较不同模型下：

```text
expected profit
approval rate
realized bad rate
average calibrated PD
KS at approval boundary
manual review rate
```

输出 profit-risk frontier：

```text
x-axis: approval rate
y-axis: expected profit or bad rate
```

## 8.7 Protocol 6：Subgroup and fairness audit

根据可用字段进行子群体评估：

```text
state
zip3 region
loan purpose
employment length group
income group if available
risk score band
```

如果使用 HMDA，则可进一步评估：

```text
race
ethnicity
sex
income
loan type
lender
```

指标：

```text
subgroup AUROC
subgroup ECE
approval rate gap
bad rate gap
equal opportunity gap
profit gap
manual review burden
```

注意：LendingClub 不一定包含法定敏感属性，不能把 geography proxy 直接等同于正式公平性结论。主文中应称为 subgroup robustness，HMDA 扩展才可讨论 fair lending audit。



## 8.8 Protocol 7：Rejected data value vs. more accepted data control

目的：证明 rejected unlabeled data 的增益不是简单来自“训练样本更多”。

设计：

```text
Base: accepted labeled training set A_train
UCRI-CS: A_train + rejected unlabeled R_train with pseudo-label weights
Control-1: A_train + same number of extra accepted samples with labels hidden and treated as unlabeled
Control-2: A_train + same number of extra accepted samples with true labels used as supervised upper bound
Control-3: A_train + randomly selected unlabeled samples with shuffled pseudo-label weights
Control-4: A_train + propensity-matched accepted samples with labels hidden, sampled so that their propensity-score distribution matches real rejected
```

比较：

```text
future accepted AUROC / PR-AUC / Brier / ECE
hidden-reject metrics under simulated rejection
decision profit ratio
pseudo-label precision-coverage
```

解释原则：

```text
若 UCRI-CS > Control-1，说明 rejected 或分布偏移样本可能提供了额外信息；
若 UCRI-CS > Control-4，才更有力说明 real rejected 数据带来超过“数量 + propensity 分布相似”的独特价值；
若 UCRI-CS ≈ Control-1，则增益主要来自半监督正则化或更多 unlabeled data，而非 rejected 特有价值；
Control-2 是有标签额外数据上限，不作为公平主 baseline。
```

## 8.9 Protocol 8：Low-label / high-rejection robustness

目的：测试当 accepted labels 较少、rejected samples 较多时，半监督方法是否更稳。

设置：

```text
retain accepted labels: 10%, 20%, 40%, 60%, 80%, 100%
rejected unlabeled pool: fixed
repeat seeds: 10
```

比较：

```text
accepted-only LightGBM
self-training without uncertainty
uncertainty-aware pseudo-labeling
calibrated distillation
full UCRI-CS
```

报告：

```text
mean ± std
performance degradation curve
calibration degradation curve
pseudo-label noise curve
```

---

## 9. Baseline 设计

## 9.1 传统 PD baseline

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
CatBoost
MLP
FT-Transformer
TabNet
SAINT
```

基线公平性要求：

```text
1. 所有强基线使用统一超参搜索预算，例如 Optuna 50-100 trials；
2. 报告搜索空间、最佳参数和 early stopping 规则；
3. TabNet/SAINT/FT-Transformer 不作为主要胜负对象，避免 strawman；
4. LightGBM/CatBoost 必须充分调优，因为它们是表格信贷任务的强基线；
5. 对传统 reject inference baseline 使用同一 shared-feature view 和同一时间切分。
```

## 9.2 Reject inference baseline

```text
Accepted-only model
Hard augmentation
Fuzzy augmentation
Parceling
Extrapolation
Self-training
Semi-supervised SVM
Propensity-weighted PD model
IPW-weighted accepted-only PD model: Stage 1 train propensity model e(x); Stage 2 train PD model on accepted samples with weights 1/e(x), without rejected pseudo-labels
Domain-adversarial accepted/rejected balancing
```

## 9.3 Pseudo-labeling and SSL baseline

```text
Vanilla pseudo-labeling
Confidence-threshold pseudo-labeling
UPS-style uncertainty-aware pseudo-labeling
FixMatch-style consistency baseline, if tabular augmentation is defensible
Mean Teacher
Noisy Student
```

对表格数据来说，强数据增强不自然，因此 FixMatch 类方法只能作为 supplementary，不应作为主线。

## 9.4 Calibration baseline

```text
No calibration
Platt scaling
Temperature scaling
Isotonic regression
Beta calibration
Ensemble calibration
Conformal prediction / conformal risk control, optional
```

## 9.5 Decision baseline

```text
Rank-only threshold by AUROC score
Calibrated PD threshold
Profit-maximizing threshold
Bad-rate-constrained threshold
Reject-option decision rule
```

---



## 9.6 Positive-Unlabeled learning baselines

拒贷推断天然可以被建模为 PU / semi-supervised problem，因此必须加入 PU baselines，避免审稿人质疑“为什么不用现成 PU learning”。

推荐构造：

```text
Positive: accepted bad/default samples
Labeled negative: accepted good/non-default samples
Unlabeled: rejected applicants
```

注意：这不是标准二类 PU，因为 accepted good 是有标签负类，rejected 是 unlabeled mixed population；因此 baseline 可设计为：

| 方法 | 说明 |
|---|---|
| uPU | unbiased PU risk estimator |
| nnPU | non-negative PU risk estimator |
| PU bagging | 从 unlabeled 中反复采样构造弱负类 |
| Elkan-Noto correction | 基于 labeled/unlabeled selection correction |
| SAR-PU / selection-aware PU | selection probability 依赖 x 的 PU 变体，若实现成本可控则加入 |
| Positive-negative-unlabeled risk | accepted good 作为可靠 negative，rejected 作为 unlabeled mixture |

对比重点不是证明 UCRI-CS “比 PU learning 名字更新”，而是证明：

```text
PU baselines 缺少 calibrated distillation 和 decision-aware objective；
PU baselines 在 covariate shift / hidden confounder / calibration 上是否退化；
UCRI-CS 的 uncertainty filtering 是否能减少 PU 伪标签污染。
```

## 9.7 Risk_Score-only baselines

为检验 UCRI-CS 是否只是复制已有信用评分，必须加入极简 baseline：

```text
Risk_Score binning: 将 Risk_Score 分箱后估计 bad rate
Risk_Score logistic regression: y ~ Risk_Score
Risk_Score + DTI LR: y ~ Risk_Score + debt_to_income_ratio
Risk_Score monotonic calibration: isotonic regression on Risk_Score
```

主文应报告 No-RiskScore setting 下 UCRI-CS 是否仍然有效；若 Risk_Score-only 接近完整模型，则论文贡献应转向“校准和决策层提升”，不能声称模型学到了大量额外违约结构。

## 9.8 Baseline 调参公平性

所有主要 baseline 使用统一搜索预算：

```text
LightGBM / CatBoost / XGBoost: Optuna 100 trials
Logistic / scorecard: regularization grid 20 trials
Neural tabular models: Optuna 50 trials + early stopping
PU baselines: prior / unlabeled weight / threshold grid 30-50 trials
Calibration methods: 固定 validation set，禁止使用 test 选择
```

TabNet、SAINT、FT-Transformer 不作为主要胜负对象；若报告，必须给出搜索空间和最佳参数，避免 strawman comparison。

## 10. 消融实验设计

## 10.1 方法模块消融

| 实验 | 目的 |
|---|---|
| Full UCRI-CS | 完整模型 |
| w/o rejected samples | 检验拒贷样本是否有贡献 |
| w/o uncertainty weighting | 检验不确定性感知是否必要 |
| hard pseudo-label instead of soft label | 检验软标签蒸馏价值 |
| uncalibrated teacher | 检验 teacher calibration 价值 |
| w/o propensity weighting | 检验选择偏差校正价值 |
| w/o representation balancing | 检验 accepted/rejected 分布对齐 |
| w/o decision loss | 检验决策优化是否改善 profit frontier |
| LightGBM student only | 检验复杂深度模型是否必要 |
| ensemble teacher vs single teacher | 检验不确定性估计质量 |

## 10.2 伪标签质量消融

报告：

```text
pseudo-label coverage
pseudo-label estimated noise
pseudo-label precision on simulated rejected group
performance vs uncertainty threshold
performance vs pseudo-label weight gamma
soft-label vs hard-label bad-rate estimation
```

关键图：

```text
x-axis: pseudo-label coverage
y-axis: pseudo-label precision / downstream AUROC / ECE
```

这能证明你的方法不是简单把所有拒贷样本都加进训练，而是在控制噪声。

## 10.3 校准消融

| 实验 | 指标 |
|---|---|
| no calibration | Brier, ECE, calibration slope |
| temperature scaling | Brier, ECE |
| isotonic regression | Brier, ECE |
| beta calibration | Brier, ECE |
| calibration before pseudo-labeling only | pseudo-label noise |
| calibration after student only | final PD reliability |
| both teacher and student calibration | full performance |

必须证明：

```text
校准不仅降低 ECE，也改善决策层 expected profit / bad-rate control。
```

## 10.4 选择偏差消融

| 实验 | 目的 |
|---|---|
| accepted-only random split | 展示乐观偏差 |
| out-of-time split | 更真实评估 |
| simulated rejection MAR | 可控验证 |
| simulated rejection MNAR-like | 更难场景 |
| IPW-weighted accepted-only | 纯选择偏差校正：只在 accepted 样本上用 1/e(x) 重加权训练 PD，不使用 rejected 伪标签 |
| SSL only | 检验半监督 |
| IPW + SSL | 检验互补性 |

## 10.5 决策层消融

比较：

```text
AUROC-optimal model
Brier-optimal model
ECE-optimal model
profit-optimal model
bad-rate-constrained model
```

展示不同优化目标下的 trade-off：

```text
高 AUROC 不一定高 profit；
低 ECE 有助于坏账率约束；
uncertainty reject option 可降低高风险自动审批错误。
```

---



## 10.6 关键稳健性消融

| 消融 | 目的 |
|---|---|
| Randomized uncertainty weights | 打乱 uncertainty score 后重新分配伪标签权重，验证不是样本数量或随机正则化带来的收益 |
| No uncertainty filtering | 所有 rejected 伪标签等权重进入训练 |
| Uncertainty component only | variance / entropy / margin 单独使用 |
| \(\tau_u\) grid sensitivity | 不同 simulated rejection 机制下阈值是否稳定 |
| Ensemble size M | M ∈ {1, 3, 5, 7, 10}，报告性能与计算开销 trade-off |
| Label definition sensitivity | Strict-matured / Early-delinquency-as-bad / Current-censored |
| LightGBM student vs FT-Transformer student | 验证深度 student 是否必要；若 LightGBM 已足够，应优先保留轻量模型 |
| More accepted unlabeled control | 排除“只是更多数据”的替代解释 |
| Confounded rejection simulation | 检验 conditional ignorability 不完全成立时的退化 |

其中 randomized uncertainty weights 是核心消融：若打乱权重后性能不下降，说明 uncertainty-aware weighting 没有提供真实信息，应重新设计权重机制。

## 11. 评价指标与统计检验

## 11.1 预测性能指标

```text
AUROC
PR-AUC
KS
F1
Recall at fixed FPR
Precision at top-K
Bad capture rate at top decile
```

信用评分中 KS 是常见业务指标，应保留。

## 11.2 校准指标

```text
Brier score
Expected Calibration Error, ECE
Maximum Calibration Error, MCE
Calibration slope
Calibration intercept
Reliability diagram
Decile-level predicted vs observed bad rate
```



### 11.2.1 ECE 固定计算协议

ECE 必须预先固定计算方式，避免事后选择有利分箱。

默认主文设置：

```text
ECE type: equal-mass / adaptive binning
Number of bins: 15
Report also: equal-width ECE with 10 and 20 bins in supplementary
Confidence interval: bootstrap 1000 resamples
```

同时报告 reliability diagram 和 calibration slope/intercept。若 equal-mass 与 equal-width 结论冲突，应以 Brier score 和 reliability diagram 辅助解释，不单独依赖某一个 ECE 数值。

### 11.2.2 PR-AUC、KS 与统计检验细则

PR-AUC 对 positive prevalence 敏感，因此每张 PR 曲线或 PR-AUC 表必须标注对应测试集 default rate，并给出 random baseline = default rate。

KS statistic 报告两项：

```text
maximum KS value
score threshold at maximum KS
```

统计检验：

```text
Main confirmatory comparisons: Holm-Bonferroni correction
Supplementary exploratory tables: Benjamini-Hochberg FDR
Effect size: Cliff's delta or paired standardized mean difference
CI: bootstrap 95% CI for AUROC, PR-AUC, Brier, ECE, Profit Ratio
```

DeLong test 只作为 AUROC 的补充，不作为多模型结论的唯一显著性依据。

## 11.3 拒贷推断特有指标

在 simulated rejection 中报告：

```text
hidden-reject AUROC
hidden-reject PR-AUC
hidden-reject ECE
pseudo-label precision
pseudo-label coverage
bad-rate estimation error
rank recovery
```

在真实 rejected 中报告：

```text
score distribution comparison
uncertainty distribution
expected bad rate of rejected group
approval boundary movement
manual review candidate analysis
```

真实 rejected 无真实 y，不报告“拒贷样本准确率”。

## 11.4 决策指标

```text
expected profit
approval rate
realized bad rate
loss rate
average loan amount approved
profit per approved loan
bad-rate constrained approval lift
manual review rate
```

建议展示：

```text
profit-risk frontier
approval-bad-rate curve
decision curve analysis
```

## 11.5 公平性与子群体指标

```text
subgroup AUROC
subgroup Brier
subgroup ECE
approval rate gap
bad rate gap
equal opportunity gap
false negative / false positive disparity
manual review burden gap
```

## 11.6 统计检验

```text
10 random seeds
mean ± standard deviation
paired Wilcoxon signed-rank test as default
bootstrap 95% CI for AUROC, PR-AUC, Brier, ECE, expected profit
DeLong test for AUROC only as supplementary
Holm-Bonferroni correction for confirmatory pairwise comparisons
Benjamini-Hochberg FDR for exploratory large-table comparisons
Bayesian signed-rank test as robustness supplement, optional
```

DeLong test 不作为唯一显著性依据。大样本下很小差异也可能显著，因此主文同时报告 effect size：

```text
ΔAUROC
ΔPR-AUC
ΔBrier
ΔECE
Δexpected profit
relative approval lift under fixed bad-rate constraint
```

---

## 12. Case study 设计

## 12.1 Case study 目标

Case study 不应只是列几个申请样本，而应围绕审批边界展示模型价值：

```text
历史策略拒绝但 UCRI-CS 认为低风险的申请人
历史策略批准但 UCRI-CS 认为高风险的申请人
accepted-only 模型过度自信但 UCRI-CS 选择人工复核的申请人
UCRI-CS 校准后改变审批决策的申请人
```

## 12.2 个案输出字段

```text
application_id
accepted/rejected indicator
historical policy score / propensity
accepted-only PD
UCRI-CS calibrated PD
uncertainty
decision
expected profit
top risk factors
top protective factors
counterfactual explanation
```

## 12.3 三类候选申请人分析

| 类型 | 定义 | 意义 |
|---|---|---|
| Type A | rejected historically, low calibrated PD, low uncertainty | 潜在误拒客户 |
| Type B | accepted historically, high calibrated PD, later defaulted | 潜在误批客户 |
| Type C | high uncertainty near threshold | 适合 manual review |

这三类分析能很好地体现 decision-aware credit scoring 的实际价值。

## 12.4 解释模板

```text
UCRI-CS assigns applicant X a calibrated PD of 0.071 and low uncertainty.
Compared with the accepted-only model, the PD decreases after incorporating rejected applicants through uncertainty-weighted distillation.
The main protective factors are low DTI, stable employment length and requested amount below peer median.
The model recommends approval under the 8% target bad-rate constraint, with positive expected profit.
```

中文：

```text
UCRI-CS 给申请人 X 的校准后违约概率为 7.1%，且不确定性较低。
相比 accepted-only 模型，引入拒贷样本的半监督蒸馏后，该申请人的风险估计下降。
主要保护性因素包括较低 DTI、较稳定的工作年限和低于同群体中位数的申请金额。
在 8% 目标坏账率约束下，模型建议批准该申请，并预计具有正向收益。
```

---



### 12.5 Case representativeness

Case study 不能只挑对 UCRI-CS 最有利的样本。每个 case 必须报告其在整体分布中的位置：

```text
risk score percentile
uncertainty percentile
overlap / out-of-support flag
whether selected from top, middle, or boundary region
comparison with baseline rank
```

建议选择三类案例：

| 类型 | 定义 | 目的 |
|---|---|---|
| High-confidence improvement | UCRI-CS 明显优于 baseline 且 uncertainty 低 | 展示方法优势 |
| Boundary applicant | 审批阈值附近、uncertainty 中等 | 展示决策校准价值 |
| Failure / disagreement case | UCRI-CS 与 Risk_Score 或 baseline 冲突，或 uncertainty 高 | 展示局限和人工复核必要性 |

这样 case study 更像方法分析，而不是选择性展示成功案例。

## 13. 推荐论文结构

### 13.1 标题备选

1. **Uncertainty-Calibrated Semi-Supervised Reject Inference for Decision-Aware Credit Scoring**
2. **Learning Calibrated Credit Scores from Accepted and Rejected Applicants via Uncertainty-Aware Semi-Supervised Reject Inference**
3. **Decision-Aware Credit Scoring under Approval Selection Bias with Uncertainty-Calibrated Reject Inference**
4. **Beyond Accepted Applicants: Calibrated Semi-Supervised Reject Inference for Credit Risk Decisioning**

首推第 1 个，简洁且与你当前方向一致。

### 13.2 摘要结构

```text
Background:
Credit scoring models are typically trained on accepted applicants only, because repayment labels are unobserved for rejected applicants. This creates approval selection bias.

Problem:
Naively assigning pseudo-labels to rejected applicants may amplify bias due to overconfident and poorly calibrated predictions.

Methods:
We propose UCRI-CS, an uncertainty-calibrated semi-supervised reject inference framework. It combines approval propensity modeling, calibrated teacher ensembles, uncertainty-aware pseudo-labeling, soft-label distillation, and decision-aware calibration.

Results:
We evaluate UCRI-CS on LendingClub accepted/rejected data and controlled simulated rejection protocols. The method is compared against accepted-only, conventional reject inference, self-training, propensity weighting, and calibration baselines.

Conclusion:
UCRI-CS improves probability calibration and decision utility while reducing pseudo-label noise, making it suitable for credit approval decision support.
```

### 13.3 正文章节

```text
1. Introduction
2. Related Work
   2.1 Credit scoring and reject inference
   2.2 Semi-supervised learning and pseudo-labeling
   2.3 Uncertainty estimation and probability calibration
   2.4 Decision-aware machine learning in credit risk
3. Data and Problem Formulation
   3.1 Accepted/rejected applicant data
   3.2 Label observability and selection bias
   3.3 Leakage-safe feature alignment
4. Method
   4.1 Approval propensity modeling
   4.2 Calibrated teacher ensemble
   4.3 Uncertainty-aware pseudo-labeling
   4.4 Soft-label distillation into lightweight student model
   4.5 Decision-aware calibration and threshold optimization
5. Experimental Design
   5.1 Out-of-time PD prediction
   5.2 Real rejected semi-supervised learning
   5.3 Simulated rejection with hidden labels
   5.4 Decision-aware approval simulation
   5.5 Subgroup and fairness audit
6. Results
   6.1 Overall comparison
   6.2 Reject inference performance
   6.3 Pseudo-label quality analysis
   6.4 Calibration analysis
   6.5 Decision utility analysis
   6.6 Ablation study
   6.7 Case studies
7. Discussion
8. Conclusion
```

---

## 14. 图表设计

### Figure 1: Problem motivation

展示：

```text
all applicants
→ historical approval policy
→ accepted applicants with observed repayment labels
→ rejected applicants with missing counterfactual labels
→ accepted-only model bias
```

核心信息：训练标签只来自被批准样本。

### Figure 2: UCRI-CS framework

展示完整 pipeline：

```text
accepted labeled data + rejected unlabeled data
→ propensity model
→ calibrated teacher ensemble
→ uncertainty-aware pseudo-labeling
→ student distillation
→ calibrated PD
→ decision layer
```

### Figure 3: Simulated rejection protocol

展示：

```text
accepted loans with known labels
→ simulate policy rejection
→ hide labels for pseudo-rejected group
→ train reject inference model
→ reveal labels for evaluation
```

这是论文可信度的关键图。

### Figure 4: Pseudo-label uncertainty analysis

展示：

```text
uncertainty threshold vs pseudo-label coverage
uncertainty threshold vs pseudo-label precision
uncertainty threshold vs downstream ECE
```

### Figure 5: Calibration results

展示：

```text
reliability diagram
ECE before/after calibration
Brier score comparison
decile predicted bad rate vs observed bad rate
```

### Figure 6: Decision utility

展示：

```text
profit-risk frontier
approval rate vs bad rate
manual review rate vs expected profit
```

### Figure 7: Case study

展示：

```text
accepted-only decision vs UCRI-CS decision
feature explanations
calibrated PD and uncertainty
expected profit
```

---

## 15. 实施计划

### 阶段 1：数据收集与标准化，4-6 周

任务：

```text
下载 LendingClub accepted/rejected 数据
解析 accepted loan_status 并构造 default label
清洗 rejected applications
对齐 accepted/rejected 共同字段
构造 shared-feature view
构造 accepted-rich view
建立时间切分
建立数据泄漏检查脚本
```

产出：

```text
UCRI-CS-LendingClub-Benchmark-v1
feature dictionary
label construction report
accepted/rejected coverage report
leakage audit report
```

### 阶段 2：baseline 与评估框架，6-8 周

任务：

```text
实现 accepted-only baseline
实现传统 reject inference baseline
实现 propensity weighting baseline
实现 vanilla self-training baseline
实现 calibration baseline
实现统一评估脚本
实现 simulated rejection protocol
```

产出：

```text
baseline leaderboard
calibration evaluation report
simulated rejection benchmark
```

### 阶段 3：UCRI-CS 模型开发，8-10 周

任务：

```text
实现 teacher ensemble
实现 uncertainty metrics
实现 teacher calibration
实现 uncertainty-aware pseudo-labeling
实现 soft-label distillation
实现 decision-aware threshold optimization
实现 reason-code explanation
```

产出：

```text
UCRI-CS model code
training pipeline
pseudo-label analysis pipeline
decision simulation pipeline
```

### 阶段 4：完整实验与消融，10-14 周

任务：

```text
accepted-only out-of-time experiment
real rejected semi-supervised experiment
simulated rejection experiment
policy shift experiment
decision utility experiment
subgroup robustness experiment
low-label robustness experiment
ablation study
```

产出：

```text
main result tables
ablation tables
calibration figures
decision utility figures
case study tables
```

**实验调度策略：**

```text
可并行：不同 random seed、不同 simulated rejection mechanism、不同 baseline 的 Optuna tuning、不同 tau_u / M / lambda sensitivity；
必须串行：数据清洗与 feature freeze → leakage audit → 主时间切分固定 → baseline tuning → UCRI-CS 主模型 → 消融与稳健性；
优先级：先完成 Protocol 1 + Protocol 3 + PU/RI baselines，再扩展到 real rejected SSL、decision utility、fairness 和 low-label robustness。
```

为了控制实验规模，主文只报告核心矩阵；rho×gamma grid、tau_u×mechanism、M sensitivity、lambda heatmap 放入 supplementary，并允许使用较少 seeds 做探索性分析，最终主结论用 10 seeds 重跑。

### 阶段 5：论文撰写、开源整理与投稿，6-8 周

任务：

```text
撰写 manuscript
整理 supplementary materials
开源代码与数据处理脚本
准备 reproducibility checklist
准备 cover letter
```

总周期预估：

```text
6-8 个月形成完整初稿；
8-10 个月达到较扎实投稿版本。
```

---



### 实验管理与复现方案

复杂实验矩阵必须从项目开始进行工程化管理：

```text
Code versioning: Git + tagged releases for each result table
Config management: YAML/ Hydra; every run stores full config
Experiment tracking: MLflow / Weights & Biases / Sacred
Random seeds: at least 5 seeds for development, 10 seeds for final tables
Data versioning: raw checksum + processed dataset version ID
Feature audit: forbidden feature checker in preprocessing pipeline
Model artifacts: save fitted calibration object, thresholds, label definition, feature list
```

每个表格结果需要能够追溯到：

```text
commit hash
data version
config file
random seed list
hardware environment
runtime and memory usage
```

### 计算资源预算

初步预算：

| 实验类型 | 资源 | 单次时间估计 | 说明 |
|---|---|---:|---|
| LightGBM/CatBoost baseline | CPU 16-32 cores, 64GB RAM | 0.5-3 h | 取决于样本量与 Optuna trials |
| FT-Transformer/MLP | 1×GPU 24GB 或 CPU fallback | 1-6 h | 神经网络不是主模型时可减少 trials |
| Ensemble teacher M=5 | CPU/GPU 混合 | baseline ×5 | 可并行 |
| Simulated rejection grid | CPU/GPU cluster preferred | 数天至数周 | 机制 × rejection rate × seed |
| Full final matrix | 多机并行 | 2-4 周墙钟时间 | 包含消融、敏感性与统计检验 |

论文中建议报告：训练时间、推理延迟、模型大小、特征数量和 calibration 开销。若无法达到真实在线部署指标，应使用 “practical/lightweight student model” 而非强称 “deployable”。

## 16. 风险评估与应对策略

| 风险 | 表现 | 应对策略 |
|---|---|---|
| 可识别性假设不清 | 审稿人质疑无法从 accepted-only 标签恢复全体 PD | 在 §3.3 明确 positivity、conditional ignorability、bounded extrapolation；只在 overlap 区域做强结论 |
| Risk_Score 循环依赖 | 模型可能只是复制已有评分卡或审批规则 | 主结论基于 No-RiskScore；报告 Input-RiskScore 与 Anchor-RiskScore 消融 |
| simulated rejection 循环验证 | 模拟机制与模型同构导致虚高 | 使用 Logistic、rule-based、score-band、geography/time、RF/MLP 多机制，并与 student 算法族解耦 |
| 真实 rejected 无标签，无法直接验证 | 审稿人质疑 reject inference 效果 | 设计 simulated rejection with hidden labels，作为可验证主实验；真实 rejected 只做分布诊断和决策分析 |
| LendingClub accepted/rejected 特征不一致 | rejected 字段较少，模型无法直接迁移 | 主实验使用 shared-feature view；accepted-rich 仅作为辅助 |
| 伪标签错误污染模型 | self-training 性能下降 | 使用 uncertainty threshold、soft labels、teacher calibration、coverage-precision 曲线 |
| 校准只改善 ECE，不改善业务价值 | 审稿人认为校准无实际意义 | 展示 bad-rate constraint、expected profit、decision curve |
| 方法被认为是模块堆叠 | teacher ensemble + calibration + SSL + decision 太多 | 全文围绕 selection bias → pseudo-label uncertainty → calibrated decision 三段主线组织 |
| 联邦学习缺少真实多机构数据 | public data 只能人为切 client | 不把 FL 放进主贡献，仅作为 supplementary non-IID extension |
| fairness 结论不足 | LendingClub 缺少法定敏感属性 | LendingClub 只做 subgroup robustness；fair lending 讨论放在 HMDA 扩展 |
| 数据泄漏 | 使用贷后字段或时间错位 | 建立 leakage audit，公开 forbidden feature list |
| expected profit 假设主观 | LGD / 收益不可精确 | 做多组 LGD、收益、资金成本敏感性分析 |
| 深度模型不如 GBDT | 表格数据常见情况 | 将 LightGBM/CatBoost 作为主学生模型，深度模型作为辅助 |

---



### 必须在 Discussion 首位承认的局限

```text
1. 真实 rejected applicants 没有还款标签，因此无法直接验证其真实 PD 校准性；
2. UCRI-CS 的强结论仅适用于 overlap 区域，不适用于历史政策从未覆盖的人群；
3. 即使 within-accepted calibration 良好，也不能推出 real rejected 区域的绝对 PD 已校准；
4. 如果审批中存在强不可观测选择因子，conditional ignorability 会被破坏，模型只能通过 confounded simulation 报告退化边界；
5. LendingClub 是平台借贷数据，不等同于银行全量信贷审批流程；rejected 文件也未必代表所有被拒或流失申请人；
6. 利润模拟简化了客户生命周期价值、竞争效应、监管资本成本和提前还款行为。
```

## 17. 最终推荐方向

建议最终论文主线固定为：

```text
UCRI-CS
= Approval selection bias modeling
+ Calibrated teacher ensemble
+ Uncertainty-aware pseudo-labeling for rejected applicants
+ Soft-label distillation into a practical lightweight credit scoring model
+ Decision-aware calibration and threshold optimization
+ Real rejected data + simulated hidden-label rejection validation
```

不要把主贡献写成：

```text
首次使用半监督学习做信用评分；
首次使用联邦学习做风控；
首次使用不确定性估计；
简单把 rejected loans 加入训练；
简单追求 AUROC 提升。
```

应写成：

```text
UCRI-CS reformulates reject inference as an uncertainty-calibrated semi-supervised learning problem under approval selection bias, and provides a decision-aware framework for learning reliable probability-of-default estimates from accepted labeled applicants and rejected unlabeled applicants.
```

中文：

> UCRI-CS 将拒贷推断重新定义为审批选择偏差下的不确定性校准半监督学习问题，通过校准教师模型、不确定性感知伪标签和软标签蒸馏，从已放款有标签样本与拒贷无标签样本中学习可用于审批决策的可靠违约概率。

最终成败取决于这几个关键点：

1. **可识别性假设和 overlap 诊断是否站得住，能否清楚限定模型适用边界；**
2. **simulated rejection 实验是否在多种外生机制下显著优于 accepted-only、传统 reject inference、PU learning、IPW-only 和 vanilla self-training；**
3. **calibration 和 uncertainty 是否真正降低 pseudo-label noise，并改善 Brier/ECE；**
4. **confounded rejection simulation 是否显示：当 conditional ignorability 被逐步破坏时，模型性能退化可被 uncertainty 捕捉；**
5. **是否诚实说明真实 rejected PD 校准性无法直接验证，只能在 overlap 区域做候选性结论；**
6. **decision-aware evaluation 是否证明模型能在坏账率约束下提升 approval / profit trade-off，并报告相对 oracle profit 的差距。**

如果这些关键点成立，该项目具备冲击 CCF B/B+ 的潜力。

---

## 18. Future work：联邦学习扩展

联邦学习不进入主项目贡献。仅在 Discussion/Future Work 中用 3-5 句话说明：若未来获得多机构真实数据，可将 UCRI-CS 扩展为 horizontal federated setting，在每个机构本地训练 teacher/student，并通过 federated distillation 或 FedAvg/FedProx 聚合。公开 LendingClub 数据只能做人为 non-IID partition，不能支撑“真实跨机构隐私保护”的强主张，因此不作为本文实验主线。

## 19. 参考文献与资料源

[1] LendingClub accepted/rejected data on Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club  
[2] LendingClub dataset on Figshare: https://figshare.com/articles/dataset/Lending_Club/22121477  
[3] Ehrhardt A. Reject inference methods in credit scoring. https://pmc.ncbi.nlm.nih.gov/articles/PMC9041715/  
[4] Kozodoi N. Shallow Self-Learning for Reject Inference in Credit Scoring. ECML-PKDD 2019 preprint.  
[5] Kiryo R, Niu G, du Plessis MC, Sugiyama M. Positive-Unlabeled Learning with Non-Negative Risk Estimator. NeurIPS 2017. https://arxiv.org/abs/1703.00593  
[6] Rizve M N, et al. In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning. https://arxiv.org/abs/2101.06329  
[7] Guo C, Pleiss G, Sun Y, Weinberger K Q. On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599  
[8] Home Credit Default Risk dataset: https://www.kaggle.com/c/home-credit-default-risk/data  
[9] CFPB HMDA public data: https://www.consumerfinance.gov/data-research/hmda/  
[10] Fannie Mae Single-Family Loan Performance Data: https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data  
[11] Freddie Mac Single-Family Loan-Level Dataset: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset  
[12] Heckman JJ. Sample selection bias as a specification error. Econometrica, 1979.  
[13] Elkan C, Noto K. Learning classifiers from only positive and unlabeled data. KDD, 2008.  
[14] Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World. Springer, 2005.  
[15] Platt JC. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. 1999.  
