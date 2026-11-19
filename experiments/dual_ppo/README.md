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

行情入口已接入上游2026年11月14日的ClickHouse及不可变快照代码，来源提交为 `ce9c647abca3ae710651290d1ff61bf3d1f3d683`。最新行情先物化为快照，再统一转为实验面板。

| 用途 | 当前安排 |
| --- | --- |
| 训练与算法对比 | 从2025年7月1日开始，使用同一批滚动任务 |
| 信息截止 | 自动查询数据库中最近已收盘的行情日期，上限为2026年12月31日 |
| 最新可评分信号 | 从实际信息截止日向前数20个交易日，保证收益标签完整 |
| 每个任务内部 | F定方向、G生成初筛、A提供上层观察、R计算奖励，各20个信号交易日 |
| 时间衔接 | 前段标签成熟后再进入后段，每10个交易日滚动，末批对齐最新成熟日 |
| 算子预热 | 自动额外取前100个交易日，默认从2025年1月27日读取行情 |
| 结果用途 | 同一批任务训练和对比，报告标记为训练期效果 |

数据库实际截止日期及任务数以准备后的面板和 `plan --cache` 输出为准。未连接数据库时，`plan` 只输出按日历估算的计划，`information_cutoff_source=calendar_plan`。后续独立检验可通过显式future任务使用新增行情。

模板含固定500只股票和399条历史公式，公式库筛选标签截至2022年11月28日。股票池历史成分与复权认证状态沿用 `exploratory`。历史主缓存的缺日记录对应旧数据，最新数据库行情需要重新核验。

## 提交到 GitHub

当前origin为 `https://github.com/FelixZhang423/QuantLab.git`，本地分支为 `codex/hierarchical-factor-search`。本次目标仓库为lcy770/QuantLabRun，当前机器已配置newrepo远端。提交包含实验源码、文档和既有历史数据。最新数据库快照与训练面板存放在var目录，按下方方式准备和迁移。

```powershell
git status --short
git add -A
git diff --cached --stat
git commit -m "Implement dual PPO design and complete comparison experiments"
git remote set-url newrepo https://github.com/lcy770/QuantLabRun.git
git push -u newrepo HEAD:main
```

仓库已包含历史主行情`backend/data_cache/master`和历史面板`datasets/ppo_reference_20260213`。`var/`下新生成的面板、模型、日志和结果按运行产物管理。数据内容、SHA256清单及历史smoke命令见 [随仓库数据集](../../datasets/README.md)。

## 实验电脑环境

在另一台电脑克隆并进入仓库，既有历史数据会随仓库取得。最新行情通过数据库快照准备。

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

## 准备最新数据库行情

首次取数需连接公司网络或VPN，并在 `backend/configs/clickhouse.local.env` 填写数据库配置。已有配置的机器直接执行准备命令。

```powershell
if (-not (Test-Path backend/configs/clickhouse.local.env)) {
    Copy-Item backend/configs/clickhouse.env.example backend/configs/clickhouse.local.env
}
notepad backend/configs/clickhouse.local.env
python -m single_factor.hierarchical.comparison_cli prepare-panel --source clickhouse --codes experiments/dual_ppo/assets/codes.json --output var/experiments/data_2025h2_2026 --training-start 2025-07-01
python -m single_factor.hierarchical.comparison_cli plan --cache var/experiments/data_2025h2_2026
```

复制模板仅在本机尚无配置时执行。`prepare-panel` 自动查询实际最新日期，保存数据库快照ID、内容指纹和真实日期范围。数据准备检查官方交易日覆盖、股票日期重复、字段与20日收益标签。完整写入后才发布面板。刷新时使用新的输出目录，并追加 `--refresh`。

准备好的面板可迁移到训练电脑，后续训练直接读取面板。

```powershell
python -m single_factor.hierarchical.comparison_cli pack-data --cache var/experiments/data_2025h2_2026 --archive var/experiments/data_2025h2_2026.zip
```

把zip放入训练电脑的var/experiments目录，再执行。

```powershell
python -m single_factor.hierarchical.comparison_cli unpack-data --cache var/experiments/data_2025h2_2026 --archive var/experiments/data_2025h2_2026.zip
```

若已有上游数据库快照，也可用 `prepare-panel --snapshot-id <ID> --snapshot-root <快照根目录>` 离线生成面板，并指定同一股票名单与新输出目录。历史主缓存导入仍可显式使用 `--source master --master <路径>`。

## 短跑验收

```powershell
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_2025h2_2026 --output var/experiments/protocol_2025h2_2026_smoke --profile smoke --device cuda
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_2025h2_2026_smoke/protocol.json --output var/experiments/results_2025h2_2026_smoke --seeds 0 --nodes g_diffusion s_upper_ppo
python -m single_factor.hierarchical.comparison_cli status --output var/experiments/results_2025h2_2026_smoke
```

smoke每个学习阶段执行一次真实更新，检查数据、训练及搜索链路。工程回归命令如下，测试中的行情为合成数据。

```powershell
python -m pytest single_factor/tests/test_comparison_database.py single_factor/tests/test_comparison_suite.py -q
```

## 三个配对种子的短训练

依次执行五组，使用同一协议与结果目录。上层奖励与配权共用一次候选池和已选成员。

```powershell
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_2025h2_2026 --output var/experiments/protocol_2025h2_2026_short --profile short --device cuda
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_2025h2_2026_short/protocol.json --output var/experiments/results_2025h2_2026_short --seeds 0 1 2 --groups generation
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_2025h2_2026_short/protocol.json --output var/experiments/results_2025h2_2026_short --seeds 0 1 2 --groups selection
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_2025h2_2026_short/protocol.json --output var/experiments/results_2025h2_2026_short --seeds 0 1 2 --groups pairing
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_2025h2_2026_short/protocol.json --output var/experiments/results_2025h2_2026_short --seeds 0 1 2 --groups reward
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_2025h2_2026_short/protocol.json --output var/experiments/results_2025h2_2026_short --seeds 0 1 2 --groups continuation weighting
python -m single_factor.hierarchical.comparison_cli summarize --output var/experiments/results_2025h2_2026_short
```

| 配置 | 结构预训练 | 下层训练 / 上层训练 / 续训预算，每节点每种子 | 每次搜索预算 |
| --- | --- | --- | --- |
| smoke | 24 公式，1 轮 | 每阶段最多 1 次更新 | 4 次尝试与 1 秒，搜索种子 0 |
| short | 144 公式，2 轮 | 120 秒 / 60 秒 / 120 秒 | 1000 次尝试与 60 秒，搜索种子 0、1、2 |
| long | 144 公式，10 轮 | 16 小时 / 8 小时 / 4 小时 | 10000 次尝试与 1800 秒，搜索种子 0、1、2 |
| primary | 144 公式，10 轮 | 48 小时 / 24 小时 / 48 小时 | 与long相同，训练期间可延后完整评价 |

尝试预算与墙钟预算分别运行并分别汇总。失败、重复与未准入的完整提案计入尝试次数。墙钟到期后完成的提案剔除。训练在完整优化批次结束处检查预算，实际耗时可能超过预算一个更新批次。预训练、候选预处理、生成训练、选择训练、续训和在线搜索分别记录。

`costs` 记录当前节点新增耗时，`cost_dependencies` 给出上游成本节点，共享预训练和生成器成本按依赖只计一次。`continuation` 是续训阶段总耗时，其中包含本阶段重生候选池的 `pool_preparation` 子项。

随机编辑、普通筛选器使用相同配对数据和随机种子重复评价。各学习组有 3 个独立训练种子，每个训练种子内有 3 个搜索重复。统计先在训练种子内平均搜索重复，再计算配对差及 bootstrap 区间。只有 3 个训练种子的区间用于描述重复实验差异。

## 当前优先安排：先训练主算法，再对比

首轮只运行主算法的种子0。新增 `primary` 配置：下层48小时，上层24小时，再交替训练48小时，合计120小时训练预算。预训练、候选池准备、检查点保存及之后的完整评价另计；预算在完整更新结束后检查，因此不是精确的五天完工时间。该预算是首轮上限，是否延长应结合学习曲线和组合表现判断。

```powershell
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_2025h2_2026 --output var/experiments/primary_w256/protocol --profile primary --device cuda --width 256
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/primary_w256/protocol/protocol.json --output var/experiments/primary_w256/results --seeds 0 --nodes c_alternating --train-only
```

依赖顺序只有 `syntax → g_diffusion → s_upper_ppo → c_alternating`，不会先训练原生RPN等对照。`--train-only` 只延后各节点训练完成后的完整搜索评价，训练所需的因子指标、准入、上层奖励和交替更新照常执行。进度中的 `phase=complete` 且 `evaluation_deferred=true` 表示本轮训练完成，评价尚未执行。

训练结束后，在相同协议与结果目录去掉 `--train-only` 即可补做评价；已保存模型会加载并跳过训练，不重新花费120小时。

```powershell
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/primary_w256/protocol/protocol.json --output var/experiments/primary_w256/results --seeds 0 --nodes c_alternating
```

确认主算法有学习效果后，再增加少数核心对照和消融，最终补齐种子1、2。基础双层与RPN加上层PPO比较使用各自基础阶段检查点；交替续训与只续下层、只续上层比较使用相同起点和相同追加预算。额外调优的主模型不能直接以更高训练预算与少训的基线作公平优劣结论。所有方法继续共用同一2025—2026年行情和任务口径，报告仍标记为训练期对比。

当前不建议省略 `--nodes` 启动全部实验组，完整实验保留为后续选项。中断后重复同一条训练命令恢复，不能直接修改已冻结协议来增加预算。

### 到达预算后继续训练

时间预算只是本轮检查点，可以在查看效果后继续追加。`extend` 继承模型权重、优化器及其动量、随机状态、候选池、历史记录和累计训练轮数；旧轮数上限不会截断新增时间。追加时间作用于所选节点的最后训练阶段：`g_diffusion` 为下层，`s_upper_ppo` 为上层，`c_alternating` 为两层交替。训练指标仍持续计算，节点 `progress.json` 的 `latest_metrics` 给出最近一次更新指标，上层更新含 `mean_return` 和 `mean_ic`。

下面从主算法上次检查点再追加24小时，结果放在新轮次目录，原模型和评价不覆盖：

```powershell
python -m single_factor.hierarchical.comparison_cli extend --protocol var/experiments/primary_w256/protocol/protocol.json --source var/experiments/primary_w256/results --output var/experiments/primary_w256/round_1 --node c_alternating --seed 0 --hours 24 --train-only
```

本轮中断后重复同一条 `extend` 命令，会继续同一预算，不会再次加24小时。下一次还想追加，则以上轮输出为来源并使用新的目录：

```powershell
python -m single_factor.hierarchical.comparison_cli extend --protocol var/experiments/primary_w256/round_1/protocol/protocol.json --source var/experiments/primary_w256/round_1/results --output var/experiments/primary_w256/round_2 --node c_alternating --seed 0 --hours 24 --train-only
```

去掉 `--train-only` 可在训练结束后完整评价；也可单独用本轮协议和results目录执行 `run --nodes c_alternating --seeds 0`。评价使用本轮模型重新计算，父轮结果不混入。每一轮保留来源协议、检查点SHA256、代码版本指纹与累计耗时。只复制必要的学习状态，不重训已经完成的依赖。支持从旧版比较实验检查点显式接续：先核验数据、奖励、网络配置和算法节点定义保持一致，建立带来源记录的新协议，随后按当前源码严格恢复；不会把旧协议冒充为当前协议。网络宽度或数据改变不属于时间续训。原起点调度旧双层的独立Campaign暂不通过此入口延长。

## 主要对比组长训练（后续可选）

根据训练期配对对比确定主要组后，在新的 long 协议执行所选节点。下面命令覆盖生成主对照、新双 PPO 与旧双 PPO，自动补齐共享预训练及依赖。

```powershell
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_2025h2_2026 --output var/experiments/protocol_2025h2_2026_long --profile long --device cuda
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_2025h2_2026_long/protocol.json --output var/experiments/results_2025h2_2026_long --seeds 0 1 2 --nodes g_rpn_original g_rpn_unified g_random g_pretrained g_diffusion g_gp s_upper_ppo p_old_dual
```

可把 `--nodes` 后的列表替换为选定节点。完整五组的长训练可执行下面一行。

```powershell
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/protocol_2025h2_2026_long/protocol.json --output var/experiments/results_2025h2_2026_long --seeds 0 1 2 --groups generation selection pairing reward continuation weighting
```

long 使用独立初始化与完整长预算，short的训练期对比用于选择需要长训练的实验组。续训机制对比由 c_lower、c_upper、c_alternating 单独实现。

### 扩大网络与容量对照

`prepare --width` 覆盖实验共用的上下层隐藏宽度，并将实际宽度写入冻结协议。省略该参数时沿用原 profile。原生自回归 RPN PPO 仍使用 profile 中独立的 `rpn_width`，当前 long 为128。扩散下层、集合选择上层及相关神经网络对照使用覆盖后的宽度。

| 隐藏宽度 | 扩散下层参数 | 集合选择上层参数 | 合计 |
| --- | ---: | ---: | ---: |
| 32 | 63,237 | 27,267 | 90,504 |
| 64 | 237,061 | 101,635 | 338,696 |
| 128 | 916,485 | 391,683 | 1,308,168 |
| 256 | 3,602,437 | 1,537,027 | 5,139,464 |

上表使用当前完整文法与4步编辑配置。参数更多不保证组合效果提高，需要在相同数据、训练种子和预算下对照。先检查扩大后的模型能否完成实际GPU更新，再启动长训练。下面先用128宽度检查自己的算法，依赖自动包含结构预训练与扩散下层；要试256时修改第一行。

```powershell
$taskWidth = 128
$taskCapacityRoot = "var/experiments/capacity_w$taskWidth"
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_2025h2_2026 --output "$taskCapacityRoot/smoke_protocol" --profile smoke --device cuda --width $taskWidth
python -m single_factor.hierarchical.comparison_cli run --protocol "$taskCapacityRoot/smoke_protocol/protocol.json" --output "$taskCapacityRoot/smoke_results" --seeds 0 --nodes p_new_dual
python -m single_factor.hierarchical.comparison_cli prepare --cache var/experiments/data_2025h2_2026 --output "$taskCapacityRoot/long_protocol" --profile long --device cuda --width $taskWidth
python -m single_factor.hierarchical.comparison_cli run --protocol "$taskCapacityRoot/long_protocol/protocol.json" --output "$taskCapacityRoot/long_results" --seeds 0 1 2 --nodes p_new_dual
```

宽度改变会改变模型张量形状，需要新协议、新结果目录并从头训练，不能接续旧宽度检查点。更新源码也会改变协议身份；先停止旧训练再更新代码，保留其原结果和源码记录。确认主算法运行后，可在同一新协议、同一结果目录再次执行 `run` 并省略 `--nodes`，补齐完整对比，已完成模型与评价会复用。GPU显存还包含优化器、梯度、数据和计算中间结果，需在训练机实际检查。

任务数N由实际数据截止日决定。long中每个生成器、每个训练种子的限时搜索预算为N×3×1800秒，另有固定尝试评价和模型训练。若形成19个任务，仅该项约28.5小时。1至2周内先完成short，再选择主要组长训。

## 断点恢复和结果

中断后重复原 `run` 命令即可恢复。模型、优化器、GP 种群、DQN 经验池、LinUCB 统计量、候选池及随机状态在每次完整更新后原子保存。完成节点会跳过。源码、配置或行情身份变化时重新准备协议并使用新的结果目录。

```powershell
python -m single_factor.hierarchical.comparison_cli status --output var/experiments/results_2025h2_2026_long
python -m single_factor.hierarchical.comparison_cli summarize --output var/experiments/results_2025h2_2026_long
```

结果目录包含 `comparison.csv`、`weight_comparison.csv`、`seed_means.json`、`paired_differences.json`，以及 `seed_<种子>/nodes/<节点>/training/` 内的逐次报告。汇总表和逐次报告都记录训练期评价用途。报告记录表达式、固定方向、成员、奖励分项、候选池身份、训练成本与逐日净值。`manifest.json` 保存环境版本、共用网络宽度 `width` 和独立的自回归宽度 `rpn_width`，`source_snapshot/` 保存本次源码。

每个真实模型目录另存 `model_metadata.json`，检查点 `checkpoint.pt` 和节点 `progress.json` 中也包含同样的 `model_metadata`。其中记录实际上下层网络类型、各自宽度、参数量、可训练参数量、权重精度、权重字节数，以及总参数量、实验名和训练种子。共享别名沿来源模型目录查看。统计只包含策略和价值神经网络，不包含优化器、DQN目标副本或LinUCB等非神经方法的系数。非神经方法标为零神经参数，原生RPN PPO按自己的实际宽度统计。原有双层和独立候选选择训练保存的模型包同样附带规模元数据；历史检查点仍可按原格式加载。

## 冻结与后续独立检验

训练期对比完成后，可冻结需要后续跟踪的方法及检查点。

```powershell
python -m single_factor.hierarchical.comparison_cli freeze --protocol var/experiments/protocol_2025h2_2026_long/protocol.json --output var/experiments/results_2025h2_2026_long --nodes p_new_dual p_new_classical p_old_dual --seeds 0 1 2 --selection var/experiments/results_2025h2_2026_long/frozen_selection.json
```

本轮默认协议全部为训练任务，冻结记录的selection_basis为training。后续独立检验需使用冻结检查点之后新增的行情及对应future任务配置。当前协议直接调用evaluate会提示缺少test任务。历史显式F/G/A/R/future协议继续支持原来的冻结验收流程。

## Top50 与配权口径

Top50 使用收盘信号、下一交易日开盘成交，每 20 个交易日调仓，初始资金 1 亿元，双边手续费率 0.0015、最低手续费 5 元、滑点 0.0005。模拟使用连续股数，尾段持仓随现有行情计价至标签成熟边界。可用停牌及涨跌停标记参与成交限制，缺失开盘价阻止成交。报告同时保存毛收益、净收益、回撤、换手和成交率。

净超额基准采用同一固定股票池的等权净收益，并使用相同交易时点与费用。Top50 收益仅用于结果评价。等权与 Ridge 固定同一个成员集合，Ridge 系数只在 F 拟合并用于后续 A/R/future。
