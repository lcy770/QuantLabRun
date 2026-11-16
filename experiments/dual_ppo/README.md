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

## 数据日期与标签成熟

本轮使用截至2026年6月16日的现有行情。使用 XSHG 官方交易日和未来 20 交易日标签。F/G/A/R 各 20 个信号交易日，相邻段等待前段全部标签成熟。训练从2024年1月2日开始，每 10 交易日滚动一次，共 27 个任务。2023年7月3日起的早期数据提供至少100个交易日的算子预热。

**2026年11月17日实际准备状态为待补8个交易日。** 主缓存最大日期为2026年6月16日，选定500只股票的日期并集仍缺2026年2月27日、3月2日至6日、3月9日及4月28日。全库5,432只股票核对3月2日和4月28日同样无记录。`prepare-panel` 已实测报告缺日，当前尚未生成本轮面板。补入这些日期的真实行情后，按下方命令准备数据及训练。开发R段的20日标签会经过这批缺日，完整划分按真实标签成熟日核验。

按官方交易日逐个检查20日标签，开发R段20个信号日全部受到整日缺失影响，保留测试31个信号日中最多12个拥有完整的市场日期标签。个别股票停牌等还可能进一步减少有效样本。核验记录保存在 `var/experiments/data_readiness_20261117.json`。本轮已完成日期划分与6项实验驱动回归，真实数据面板及训练等待源行情补齐。

按同一成熟等待规则，训练若从2025年开始仅有2个完整任务，因此本轮将训练起点前移一年。开发与保留测试使用最新的2026年数据。

| 用途 | F | G | A | R 或独立信号段 | 信息截止 |
| --- | --- | --- | --- | --- | --- |
| 训练 | 从 2024-01-02 开始滚动 | 同任务顺序安排 | 同任务顺序安排 | 同任务顺序安排 | 训练标签最晚截止上限 2025-09-24 |
| 开发选模 | 2025-07-31～08-27 | 2025-09-25～10-30 | 2025-11-28～12-25 | R 为 2026-01-27～03-03 | R 标签在 2026-03-31 全部成熟 |
| 保留测试 | 2025-09-25～10-30 | 2025-11-28～12-25 | 2026-01-27～03-03 | future 为 2026-04-01～05-19 | 2026-03-31 冻结，标签在 06-16 全部成熟 |

训练反馈标签严格早于开发 G 的开始。开发 F 可使用此前历史信息。独立验收先用 F/G/A 生成候选并选定成员，再打开 future 标签。

当前主行情缓存位于 `backend/data_cache/master`，最大记录日期为 **2026-06-16**，整日缺失状态见上文。旧PPO面板截至2026-02-13，本轮目标面板为 `var/experiments/data_through_20260616`。`prepare` 会检查覆盖范围和官方交易日。最后20个交易日仅为先前信号提供成熟标签，最晚可评价信号为2026-05-19。

本轮保留现有股票池、复权及数据约定的 `exploratory` 状态。保留测试约一个半月，完整结果按该时长解释。

模板内的 500 只股票名单与 399 条历史公式库固定并随 Git 迁移。公式库筛选标签截至 2022-11-28。股票名单沿用现有研究股票池，其历史退市覆盖、成分股时点与复权口径验证状态由真实数据清单记录，默认 `data_mode=exploratory`。

## 提交到 GitHub

当前远端为 `https://github.com/FelixZhang423/QuantLab.git`，当前分支为 `codex/hierarchical-factor-search`。在当前电脑执行以下命令可将整个工作区源码及小型实验资产提交到此分支。

```powershell
git status --short
git add -A
git diff --cached --stat
git commit -m "Implement dual PPO design and complete comparison experiments"
git push -u origin codex/hierarchical-factor-search
```

`var/` 下的行情、模型、日志和结果由 `.gitignore` 忽略。市场数据用下述数据包迁移。

## 实验电脑环境

在另一台电脑克隆并进入仓库。

```powershell
git clone --branch codex/hierarchical-factor-search https://github.com/FelixZhang423/QuantLab.git
cd QuantLab
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

根据开发结果确定主要组后，在新的 long 协议执行所选节点。下面命令覆盖生成主对照、新双 PPO 与旧双 PPO，自动补齐共享预训练及依赖。

```powershell
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_through_20260616 --output var/experiments/protocol_20260616_long --profile long --device cuda
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_long/protocol.json --output var/experiments/results_20260616_long --seeds 0 1 2 --nodes g_rpn_original g_rpn_unified g_random g_pretrained g_diffusion g_gp s_upper_ppo p_old_dual
```

可把 `--nodes` 后的列表替换为选定节点。完整五组的长训练可执行下面一行。

```powershell
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_20260616_long/protocol.json --output var/experiments/results_20260616_long --seeds 0 1 2 --groups generation selection pairing reward continuation weighting
```

long 使用独立初始化与完整长预算，short 结果用于选择需要长训练的实验组。续训机制对比由 c_lower、c_upper、c_alternating 单独实现。

## 断点恢复和结果

中断后重复原 `run` 命令即可恢复。模型、优化器、GP 种群、DQN 经验池、LinUCB 统计量、候选池及随机状态在每次完整更新后原子保存。完成节点会跳过。源码、配置或行情身份变化时重新准备协议并使用新的结果目录。

```powershell
python -m single_factor.hierarchical.comparison_cli status --output var/experiments/results_20260616_long
python -m single_factor.hierarchical.comparison_cli summarize --output var/experiments/results_20260616_long
```

结果目录包含 `comparison.csv`、`weight_comparison.csv`、`seed_means.json`、`paired_differences.json`，以及 `seed_<种子>/nodes/<节点>/development/` 内的逐次报告。报告记录表达式、固定方向、成员、奖励分项、候选池身份、训练成本与逐日净值。`manifest.json` 保存环境版本，`source_snapshot/` 保存本次源码。

## 冻结后独立验收

下方给出新双 PPO、新下层加常规筛选、旧双 PPO 的冻结命令。将节点列表改为开发结果确定的最终方法后执行。freeze 固定节点、训练种子及检查点哈希，evaluate 校验冻结身份并记录测试数据访问。

```powershell
python -m single_factor.hierarchical.comparison_cli freeze --protocol var/experiments/protocol_20260616_long/protocol.json --output var/experiments/results_20260616_long --nodes p_new_dual p_new_classical p_old_dual --seeds 0 1 2 --selection var/experiments/results_20260616_long/frozen_selection.json
python -m single_factor.hierarchical.comparison_cli evaluate --protocol var/experiments/protocol_20260616_long/protocol.json --output var/experiments/results_20260616_long --selection var/experiments/results_20260616_long/frozen_selection.json
python -m single_factor.hierarchical.comparison_cli summarize --output var/experiments/results_20260616_long --split test
```

验收报告位于各节点的 `test/`，汇总输出以 `test_` 开头。

## Top50 与配权口径

Top50 使用收盘信号、下一交易日开盘成交，每 20 个交易日调仓，初始资金 1 亿元，双边手续费率 0.0015、最低手续费 5 元、滑点 0.0005。模拟使用连续股数，尾段持仓随现有行情计价至标签成熟边界。可用停牌及涨跌停标记参与成交限制，缺失开盘价阻止成交。报告同时保存毛收益、净收益、回撤、换手和成交率。

净超额基准采用同一固定股票池的等权净收益，并使用相同交易时点与费用。Top50 收益仅用于结果评价。等权与 Ridge 固定同一个成员集合，Ridge 系数只在 F 拟合并用于后续 A/R/future。
