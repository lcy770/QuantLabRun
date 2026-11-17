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

公司Windows训练电脑已确认有Python 3.11.7、RTX 4060和610.60驱动，可按本节末尾的“现有Python与并发下载”操作。

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

### 现有Python与并发下载

当前公司电脑的Conda源连接失败、winget源缺失、aria2安装包下载超时。可用已有Python的标准库下载入口，默认8连接，按16MiB分段缓存，重试时续传，并核验PyTorch官方SHA256。总速度受公司网络出口和代理影响，以DL和ETA输出为准。

在公司电脑的项目目录执行。首次使用时先创建实验环境，已有环境时直接执行下载命令。

```powershell
Set-Location 'C:\Users\13650\QuantLabRun-train'
git -c http.version=HTTP/1.1 pull --ff-only
& 'D:\Anaconda3\python.exe' -m venv 'var\venvs\dualppo'
& 'D:\Anaconda3\python.exe' -m single_factor.download_runtime --workers 8
```

下载完成会输出 `Verified download`，已完成文件再次运行会输出 `Verified existing file`。Ctrl+C后重跑相同命令可以续传。连接中断时保留已保存分段，校验失败时清理损坏分段。该入口只下载当前Windows、Python3.11所需的Torch 2.11.0+cu128安装包。

确认下载校验成功后，依次安装GPU版Torch、其余实验依赖，并检查设备。每条安装成功后执行下一条。

```powershell
$taskPython = Join-Path $PWD 'var\venvs\dualppo\Scripts\python.exe'
& $taskPython -m pip install 'var/wheels/torch-2.11.0+cu128-cp311-cp311-win_amd64.whl' --index-url https://pypi.tuna.tsinghua.edu.cn/simple --retries 10 --timeout 120
& $taskPython -m pip install -r requirements-experiments.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --retries 10 --timeout 120
& $taskPython -m single_factor.hierarchical.comparison_cli doctor
```

本节后续示例中的 `python -m`，在这个虚拟环境中执行时使用 `& $taskPython -m`。下载器回归为 `python -m pytest single_factor/tests/test_download_runtime.py -q`，使用本地合成HTTP服务验证，真实源的8连接小范围读取另有核验。

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

尝试预算与墙钟预算分别运行并分别汇总。失败、重复与未准入的完整提案计入尝试次数。墙钟到期后完成的提案剔除。训练在完整优化批次结束处检查预算，实际耗时可能超过预算一个更新批次。预训练、候选预处理、生成训练、选择训练、续训和在线搜索分别记录。

`costs` 记录当前节点新增耗时，`cost_dependencies` 给出上游成本节点，共享预训练和生成器成本按依赖只计一次。`continuation` 是续训阶段总耗时，其中包含本阶段重生候选池的 `pool_preparation` 子项。

随机编辑、普通筛选器使用相同配对数据和随机种子重复评价。各学习组有 3 个独立训练种子，每个训练种子内有 3 个搜索重复。统计先在训练种子内平均搜索重复，再计算配对差及 bootstrap 区间。只有 3 个训练种子的区间用于描述重复实验差异。

## 主要对比组长训练

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

任务数N由实际数据截止日决定。long中每个生成器、每个训练种子的限时搜索预算为N×3×1800秒，另有固定尝试评价和模型训练。若形成19个任务，仅该项约28.5小时。1至2周内先完成short，再选择主要组长训。

## 断点恢复和结果

中断后重复原 `run` 命令即可恢复。模型、优化器、GP 种群、DQN 经验池、LinUCB 统计量、候选池及随机状态在每次完整更新后原子保存。完成节点会跳过。源码、配置或行情身份变化时重新准备协议并使用新的结果目录。

```powershell
python -m single_factor.hierarchical.comparison_cli status --output var/experiments/results_2025h2_2026_long
python -m single_factor.hierarchical.comparison_cli summarize --output var/experiments/results_2025h2_2026_long
```

结果目录包含 `comparison.csv`、`weight_comparison.csv`、`seed_means.json`、`paired_differences.json`，以及 `seed_<种子>/nodes/<节点>/training/` 内的逐次报告。汇总表和逐次报告都记录训练期评价用途。报告记录表达式、固定方向、成员、奖励分项、候选池身份、训练成本与逐日净值。`manifest.json` 保存环境版本，`source_snapshot/` 保存本次源码。

## 冻结与后续独立检验

训练期对比完成后，可冻结需要后续跟踪的方法及检查点。

```powershell
python -m single_factor.hierarchical.comparison_cli freeze --protocol var/experiments/protocol_2025h2_2026_long/protocol.json --output var/experiments/results_2025h2_2026_long --nodes p_new_dual p_new_classical p_old_dual --seeds 0 1 2 --selection var/experiments/results_2025h2_2026_long/frozen_selection.json
```

本轮默认协议全部为训练任务，冻结记录的selection_basis为training。后续独立检验需使用冻结检查点之后新增的行情及对应future任务配置。当前协议直接调用evaluate会提示缺少test任务。历史显式F/G/A/R/future协议继续支持原来的冻结验收流程。

## Top50 与配权口径

Top50 使用收盘信号、下一交易日开盘成交，每 20 个交易日调仓，初始资金 1 亿元，双边手续费率 0.0015、最低手续费 5 元、滑点 0.0005。模拟使用连续股数，尾段持仓随现有行情计价至标签成熟边界。可用停牌及涨跌停标记参与成交限制，缺失开盘价阻止成交。报告同时保存毛收益、净收益、回撤、换手和成交率。

净超额基准采用同一固定股票池的等权净收益，并使用相同交易时点与费用。Top50 收益仅用于结果评价。等权与 Ridge 固定同一个成员集合，Ridge 系数只在 F 拟合并用于后续 A/R/future。
