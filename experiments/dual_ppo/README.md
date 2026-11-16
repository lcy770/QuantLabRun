# 双层 PPO 对比实验与跨机运行

本目录提供五组实验的可迁移模板、公共公式库和股票名单。所有命令在仓库根目录运行。Windows PowerShell 与 Linux shell 均可执行下方逐行命令。

## 有效实验方案

按 2026-11-16 的确认，上层完整奖励继续采用三项，数量项消融已删除。原始方案保存在 [experiment_plan.md](experiment_plan.md)。

\[
r_{\mathrm{lower}}=Q(f)-\kappa\frac{t(f)}{t_{\mathrm{ref}}}+\eta D(f,S),\qquad
U(S)=\alpha\overline{\mathrm{IC}}(S)-\beta\sigma_{\mathrm{block}}(S)-\gamma H(S).
\]

方向只在 F 固定，Q 和四项准入在 G 计算，上层输入只来自 A，训练终点奖励在 R 计算。H 默认使用奖励段的成员缺陷均值，D 使用其他成员固定、移除一个成员后重新等权的效用差。默认 η=0，α=50、β=25、γ=0.5、κ=0.05。公共门槛为 IC≥0.02、RankIC≥0.01、ICIR≥0.1、覆盖率≥0.8。实现复用主算法的质量、准入、信号标准化及奖励函数。

| 组 | 节点 | 实际实现 |
| --- | --- | --- |
| generation | g_rpn_original / g_rpn_unified | 原 LSTM 自回归 RPN MaskablePPO，分别使用原绝对指标奖励与统一 Q−κC |
| generation | g_random / g_pretrained | 随机合法原生 RPN 编辑 / 只进行结构预训练的扩散编辑器 |
| generation | g_diffusion / g_gp | 新下层起点与扩散 PPO / 锦标赛、合法子式交叉、变异、精英保留 GP |
| selection | s_random / s_quality / s_forward_exchange | 同一冻结候选池上的随机、A 段 IC 排名、前向加入加一换一 |
| selection | s_linucb / s_double_dqn / s_upper_ppo | 同池 LinUCB、Double DQN、集合 PPO |
| pairing | p_rpn / p_random / p_gp | 上层 PPO 分别接原 RPN、随机编辑、GP |
| pairing | p_old_dual / p_new_classical / p_new_dual | 原起点调度双 PPO / 新下层加前向交换 / 新双 PPO |
| reward | a_full / a_no_stability / a_no_quality | 三项完整奖励 / β=0 / γ=0 |
| continuation | c_lower / c_upper / c_alternating | 从同一新双 PPO 检查点开始，仅续下层 / 仅续上层 / 下层更新后重生候选再训上层 |
| weighting | w_equal_ridge | 同一已选集合，比较等权与只在 F 拟合并锁定的 Ridge |

生成器组统一使用 A 段前向交换筛选。上层组与奖励消融复用同一生成器检查点、训练候选池和评价候选池。各消融按各自奖励训练，最终统一按三项完整奖励评价。原双 PPO 调用保留的旧训练和搜索实现，保留原拟合期奖励、起点调度与 Ridge 工作池。

p_new_classical 与 g_diffusion、p_new_dual 与 s_upper_ppo、a_full 与 s_upper_ppo、w_equal_ridge 与 s_upper_ppo 共享真实模型。依赖自动补齐，跨组调用可复用已完成节点。

## 当前数据与训练对比口径

本轮以2026年6月16日为信息截止日，训练与算法对比共用同一批历史滚动任务。默认任务从2024年1月2日开始，共45个任务。20交易日收益标签最晚支持2026年5月19日的信号，最后一批任务的标签在6月16日成熟。2023年7月3日起的数据为算子提供至少100个交易日的预热。

| 用途 | 当前安排 |
| --- | --- |
| 训练与算法对比 | 同一组45个滚动任务，全部方法使用相同日期、股票池、门槛和配对预算 |
| 每个任务内部 | F定方向、G生成和初筛、A提供上层观察、R计算终点奖励，各20个信号交易日 |
| 时间衔接 | 前一段20日标签成熟后进入下一段，每10个交易日滚动，末批额外对齐最新成熟日 |
| 结果含义 | 训练期对比，报告标记evaluation_scope=training，evaluation_uses_training_data=true |
| 后续独立检验 | 使用2026年6月16日之后新增的行情，届时配置新的验收任务 |

最新任务的F为2025年10月20日至11月14日，G为2025年12月15日至2026年1月13日，A为2026年2月11日至3月18日，R为2026年4月17日至5月19日。任务内部的A是上层可见观察段，R是训练奖励段。

**2026年11月17日数据准备状态为待补8个交易日。** 选定500只股票的日期并集缺2026年2月27日、3月2日至6日、3月9日及4月28日。全库5,432只股票核对3月2日和4月28日同样无记录。主缓存最大日期为2026年6月16日，`prepare-panel`已实测报告缺日，目标面板尚未生成。补入这8日真实行情后，按下方命令准备数据。核验记录保存在`var/experiments/data_readiness_20261117.json`。

主缓存位于`backend/data_cache/master`，目标面板为`var/experiments/data_through_20260616`。旧PPO面板截至2026年2月13日。本轮沿用现有股票池、复权及数据约定的`exploratory`状态。

模板内500只股票名单和399条历史公式随Git迁移。公式库筛选标签截至2022年11月28日。股票池历史退市覆盖、成分股时点和复权口径的验证状态保存在真实数据清单中。

## 提交到 GitHub

当前origin为 `https://github.com/FelixZhang423/QuantLab.git`，本地分支为 `codex/hierarchical-factor-search`。本次目标仓库为lcy770/QuantLabRun，当前机器已配置newrepo远端。提交包含源码、文档、主行情和历史实验面板。

```powershell
git status --short
git add -A
git diff --cached --stat
git commit -m "Implement dual PPO design and complete comparison experiments"
git remote set-url newrepo https://github.com/lcy770/QuantLabRun.git
git push -u newrepo HEAD:main
```

主行情`backend/data_cache/master`和历史面板`datasets/ppo_reference_20260213`随Git提交。`var/`下新生成的面板、模型、日志和结果按运行产物管理。数据内容、SHA256清单及历史smoke命令见 [随仓库数据集](../../datasets/README.md)。

## 实验电脑环境

在另一台电脑克隆并进入仓库，主行情和历史面板会一并检出。

```powershell
git clone --branch main https://github.com/lcy770/QuantLabRun.git
cd QuantLabRun
conda create -n dualppo python=3.11 -y
conda activate dualppo
python -m pip install --upgrade pip
```

先按实验机驱动安装 PyTorch 的 CUDA 构建。下面给出 CUDA 12.8 构建的命令，其他构建可从 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 选择。

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-experiments.txt
python -m single_factor.hierarchical.comparison_cli doctor
```

`doctor` 应显示 `cuda_available=true` 和实验机 GPU。`requirements-experiments.txt` 同时安装本仓库的 backend 与 quantops，以便从干净克隆导入公共算子和表达式适配器。CPU 工程验收可使用普通 PyTorch 构建，并在 `prepare` 时指定 `--device cpu`。

## 准备并迁移行情

数据源是 `master/stocks/<股票代码>.parquet`，列包含 `date`、`ycp`、`open`、`close`、`high`、`low`、`volume`、`amount`、`TurnoverRate`。停牌、涨停和跌停标记存在时一并保留。当前电脑使用下方实际主缓存路径，另一台电脑按复制后的路径设置 `--master`。

```powershell
python -m single_factor.hierarchical.comparison_cli prepare-panel --master backend/data_cache/master --codes experiments/dual_ppo/assets/codes.json --output var/experiments/data_through_20260616 --start 2023-07-03 --end 2026-06-16
python -m single_factor.hierarchical.comparison_cli pack-data --cache var/experiments/data_through_20260616 --archive var/experiments/data_through_20260616.zip
```

把 zip 复制到实验电脑的 `var/experiments/`，在实验电脑执行。

```powershell
python -m single_factor.hierarchical.comparison_cli unpack-data --cache var/experiments/data_through_20260616 --archive var/experiments/data_through_20260616.zip
```

解包要求目标目录为空，并逐文件验证 SHA256。也可把完整 master 复制到实验电脑后，直接在该机执行 `prepare-panel`。

数据准备采用因果前收盘参考价收益链，未来标签要求接下来的 20 个官方交易日均有效。代码拒绝缺失整日市场行情，不补造行情。`prepare` 在实验电脑重新解析绝对路径并核验数据身份。

## 短跑验收

```powershell
python -m single_factor.hierarchical.comparison_cli plan
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_through_20260616 --output var/experiments/protocol_20260616_smoke --profile smoke --device cuda
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_smoke/protocol.json --output var/experiments/results_20260616_smoke --seeds 0
python -m single_factor.hierarchical.comparison_cli status --output var/experiments/results_20260616_smoke
```

smoke 对每个学习器执行一次真实更新，覆盖五组实验，只有一个训练种子和一个搜索重复。1 秒搜索窗口可能因公式库预处理耗时返回空池，空集合奖励为零、组合持现金。

当前电脑可直接执行包含合成行情的完整工程验收，覆盖真实自回归 PPO、GP、LinUCB、Double DQN、集合 PPO、旧双 PPO、消融、续训、恢复和冻结验收。

```powershell
python -m pytest single_factor/tests/test_comparison_suite.py -q
```

主缓存补齐后，本机先运行下方真实数据准备与CPU短流程。若前面已经生成面板，直接从 `prepare` 开始。

```powershell
$taskPython = 'C:/Users/Sasuke/miniconda3/envs/alphagen/python.exe'
& $taskPython -m single_factor.hierarchical.comparison_cli prepare-panel --master backend/data_cache/master --output var/experiments/data_through_20260616
& $taskPython -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_through_20260616 --output var/experiments/protocol_20260616_smoke_cpu --profile smoke --device cpu
& $taskPython -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_smoke_cpu/protocol.json --output var/experiments/results_20260616_smoke_cpu --seeds 0 --nodes g_diffusion s_upper_ppo
```

## 三个配对种子的短训练

依次执行五组，使用同一协议与结果目录。上层奖励与配权共用一次候选池和已选成员。

```powershell
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_through_20260616 --output var/experiments/protocol_20260616_short --profile short --device cuda
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_short/protocol.json --output var/experiments/results_20260616_short --seeds 0 1 2 --groups generation
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_short/protocol.json --output var/experiments/results_20260616_short --seeds 0 1 2 --groups selection
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_short/protocol.json --output var/experiments/results_20260616_short --seeds 0 1 2 --groups pairing
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_short/protocol.json --output var/experiments/results_20260616_short --seeds 0 1 2 --groups reward
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_short/protocol.json --output var/experiments/results_20260616_short --seeds 0 1 2 --groups continuation weighting
python -m single_factor.hierarchical.comparison_cli summarize --output var/experiments/results_20260616_short
```

| 配置 | 结构预训练 | 下层训练 / 上层训练 / 续训预算，每节点每种子 | 每次搜索预算 |
| --- | --- | --- | --- |
| smoke | 24 公式，1 轮 | 每阶段最多 1 次更新 | 4 次尝试与 1 秒，搜索种子 0 |
| short | 144 公式，2 轮 | 120 秒 / 60 秒 / 120 秒 | 1000 次尝试与 60 秒，搜索种子 0、1、2 |
| long | 144 公式，10 轮 | 16 小时 / 8 小时 / 4 小时 | 10000 次尝试与 1800 秒，搜索种子 0、1、2 |

尝试预算与墙钟预算分别运行并分别汇总。失败、重复与未准入的完整提案计入尝试次数。墙钟到期后完成的提案剔除。训练在完整优化批次结束处检查预算，实际耗时可能超过预算一个更新批次。预训练、候选预处理、生成训练、选择训练、续训和在线搜索分别记录。

`costs` 记录当前节点新增耗时，`cost_dependencies` 给出上游成本节点，共享预训练和生成器成本按依赖只计一次。`continuation` 是续训阶段总耗时，其中包含本阶段重生候选池的 `pool_preparation` 子项。

随机编辑、普通筛选器使用相同配对数据和随机种子重复评价。各学习组有 3 个独立训练种子，每个训练种子内有 3 个搜索重复。统计先在训练种子内平均搜索重复，再计算配对差及 bootstrap 区间。只有 3 个训练种子的区间用于描述重复实验差异。

## 主要对比组长训练

根据训练期配对对比确定主要组后，在新的 long 协议执行所选节点。下面命令覆盖生成主对照、新双 PPO 与旧双 PPO，自动补齐共享预训练及依赖。

```powershell
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_through_20260616 --output var/experiments/protocol_20260616_long --profile long --device cuda
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_long/protocol.json --output var/experiments/results_20260616_long --seeds 0 1 2 --nodes g_rpn_original g_rpn_unified g_random g_pretrained g_diffusion g_gp s_upper_ppo p_old_dual
```

可把 `--nodes` 后的列表替换为选定节点。完整五组的长训练可执行下面一行。

```powershell
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_long/protocol.json --output var/experiments/results_20260616_long --seeds 0 1 2 --groups generation selection pairing reward continuation weighting
```

long 使用独立初始化与完整长预算，short的训练期对比用于选择需要长训练的实验组。续训机制对比由 c_lower、c_upper、c_alternating 单独实现。

45个任务的全部限时搜索也计入总成本。long中每个生成器、每个训练种子的限时搜索预算为45×3×1800秒，约67.5小时，另有固定尝试评价和模型训练。完整五组的全量长训加评价会超过1至2周，执行范围依据short结果和实际时间预算确定。

## 断点恢复和结果

中断后重复原 `run` 命令即可恢复。模型、优化器、GP 种群、DQN 经验池、LinUCB 统计量、候选池及随机状态在每次完整更新后原子保存。完成节点会跳过。源码、配置或行情身份变化时重新准备协议并使用新的结果目录。

```powershell
python -m single_factor.hierarchical.comparison_cli status --output var/experiments/results_20260616_long
python -m single_factor.hierarchical.comparison_cli summarize --output var/experiments/results_20260616_long
```

结果目录包含 `comparison.csv`、`weight_comparison.csv`、`seed_means.json`、`paired_differences.json`，以及 `seed_<种子>/nodes/<节点>/training/` 内的逐次报告。汇总表和逐次报告都记录训练期评价用途。报告记录表达式、固定方向、成员、奖励分项、候选池身份、训练成本与逐日净值。`manifest.json` 保存环境版本，`source_snapshot/` 保存本次源码。

## 冻结与后续独立检验

训练期对比完成后，可冻结需要后续跟踪的方法及检查点。

```powershell
python -m single_factor.hierarchical.comparison_cli freeze --protocol var/experiments/protocol_20260616_long/protocol.json --output var/experiments/results_20260616_long --nodes p_new_dual p_new_classical p_old_dual --seeds 0 1 2 --selection var/experiments/results_20260616_long/frozen_selection.json
```

本轮默认协议全部为训练任务，冻结记录的selection_basis为training。后续独立检验需要6月16日之后的行情及对应future任务配置。当前协议直接调用evaluate会提示缺少test任务。历史显式F/G/A/R/future协议继续支持原来的冻结验收流程。

## Top50 与配权口径

Top50 使用收盘信号、下一交易日开盘成交，每 20 个交易日调仓，初始资金 1 亿元，双边手续费率 0.0015、最低手续费 5 元、滑点 0.0005。模拟使用连续股数，尾段持仓随现有行情计价至标签成熟边界。可用停牌及涨跌停标记参与成交限制，缺失开盘价阻止成交。报告同时保存毛收益、净收益、回撤、换手和成交率。

净超额基准采用同一固定股票池的等权净收益，并使用相同交易时点与费用。Top50 收益仅用于结果评价。等权与 Ridge 固定同一个成员集合，Ridge 系数只在 F 拟合并用于后续 A/R/future。
