# 随仓库提供的数据集

本次按用户授权将现有行情和历史实验面板提交到 `lcy770/QuantLabRun`，克隆仓库即可取得文件。`manifest.json`记录各数据文件的大小和SHA256。

| 数据 | 路径 | 范围 |
|---|---|---|
| 主行情 | `backend/data_cache/master` | 5,432只股票，2021年6月18日至2026年6月16日，约836MB |
| 历史实验面板 | `datasets/ppo_reference_20260213` | 500只股票，截至2026年2月13日，7个特征及20日收益标签，约29.5MB |
| 当前数据核验 | `datasets/data_readiness_20261117.json` | 45个训练对比任务的日期信息和8个整日缺失记录 |

主行情缺2026年2月27日、3月2日至6日、3月9日及4月28日。最大记录日期为6月16日，完整最新面板的准备仍需补这8日真实行情。历史面板中的2025年部分换手率缺失已经标记为NaN。现有数据约定的认证状态为exploratory。

## 克隆后检查历史流程

安装项目实验依赖后，在仓库根目录执行。该流程使用历史任务，smoke配置只进行少量更新以检查代码、数据和模型连接。

```powershell
python -m single_factor.hierarchical.comparison_cli prepare --cache datasets/ppo_reference_20260213 --tasks-config single_factor/hierarchical_config.example.json --output var/experiments/bundled_smoke_protocol --profile smoke --device cpu
python -m single_factor.hierarchical.comparison_cli run --protocol var/experiments/bundled_smoke_protocol/protocol.json --output var/experiments/bundled_smoke_results --seeds 0 --nodes g_diffusion s_upper_ppo
python -m single_factor.hierarchical.comparison_cli summarize --output var/experiments/bundled_smoke_results
```

最新45任务的方案、长短训练及GPU命令见 [实验运行指南](../experiments/dual_ppo/README.md)。本目录数据与主行情按原始字节存储，数据读取路径由配置相对位置解析。
