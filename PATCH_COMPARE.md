# Patch 对比与两个仓库的 LIFT 核对

## 固定参照

- 本次修改前 lift-test：`400a1c856b89882f89b5ca444b2400ed658be424`。
- 基础模型 master：[`352c88613ee0f4d9597591b2e49ccce4a490e2ae`](https://github.com/Capybaring/exo-base-model/tree/352c88613ee0f4d9597591b2e49ccce4a490e2ae)。
- 官方 LIFT：[`79cf8f157bb9a616c9733daf1e9c2cfefdc98c94`](https://github.com/SJTU-DMTai/LIFT/tree/79cf8f157bb9a616c9733daf1e9c2cfefdc98c94)。

## 运行

安装现有 requirements.txt 后，在 lift-test 目录执行：

```bash
python compare_patch.py \
  --data-dir ../../DAG/dataset/forecasting/ \
  --datasets BE NP DE PJM FR \
  --patch-hours 12 \
  --output ./comparison_results
```

只处理 BE：`--datasets BE`。默认即处理全部五个数据集。支持 data-dir 指向 CSV 所在目录，或包含 forecasting/ 的上一级。

复核 1 h patch 等价原始小时值：

```bash
python compare_patch.py --data-dir ../../DAG/dataset/forecasting/ --patch-hours 1 --output ./comparison_1h
```

`--window-stride-hours` 默认 168，控制相邻历史样本起点；`--patch-stride-hours` 默认等于 patch 宽度，控制同一历史窗口内的 patch 起点。两个步长不是同一参数。输出已存在时换目录或加 `--overwrite`。支持整数小时宽度、重叠 patch 和裁短的尾部 patch；宽度不超过未来区间 24 h，与基础 collator 的限制一致。

## 如何与基础模型划分一致

读取基础仓库 `exobase/data.py::pack_patches` 和 `exobase/model.py::scalar_patch_means`：

- 固定时长、左闭右开区间，锚点为每个历史窗口起点；按时间归属分块，不按不规则观测数量分块。
- 设窗口时长 T、宽度 W、patch 步长 S，块数为 `ceil(max(T-W,0)/S)+1`，第 p 块为 `[p*S, min(p*S+W,T))`。
- 每块各变量的观测值取算术均值；本任务规则完整小时数据不需要缺失填补、TTCN、插值或重采样。
- 168 h / 12 h 得到 14 个均值，分别对应 `[0,12)`、`[12,24)`、……、`[156,168)`。

基础仓库当前提交的 NP/BE 及 table3 短期配置是 `patch_seconds=3600`（1 h），不是 12 h。按用户明确确认，本实验使用 12 h，复用其边界规则和均值摘要方式。基础模型的 patch-lift 滞后诊断已经使用标量 patch 均值，见 `exobase/y_forecast.py::estimate_lags`；神经预测主路径另用 TTCN，本实验不运行预测网络。

## 两者是不是官方循环 LIFT

| 对象 | 核对结论 |
| --- | --- |
| lift-test 原始估计器 | 窗口内标准化、长度 L 的循环 FFT、绝对值局部峰和 1..L-2 候选与固定官方 efficient 实现一致，只计算 X→OT 并保留两个外生 |
| exo-base-model 默认 patch-lift 的 lag_estimator=fft | 标量 patch 均值送入 `estimate_fft_leads`，完整、非退化输入下与官方标量循环公式一致；缺失、常量、空通道有额外有效性处理 |
| exo-base-model 的 lag_estimator=pearson | 显式非循环选项，不是默认 FFT 模式；旧 checkpoint 可能恢复该模式 |
| 整个基础预测模型 | 包含不规则分块、训练集缩放、编码器、X→Y 约束及可训练特征支路，不能称为官方完整预测网络的逐项原样复现 |

本比较沿用 lift-test 的原量纲 float32 输入，不引入基础训练管线的训练集归一化。均值后执行相同窗口归一化。全局仿射缩放通常被窗口中心化/标准化消除，但 epsilon=1e-8 和浮点运算会影响极低方差情况，不能保证任意完整训练流程逐位一致。

常量序列方面，lift-test 保留官方 lag=1、score=0 回退并标记；基础模型会将无效通道置 lag=0 并排除。这是边界处理区别，不应当把回退的 1 个 patch 解释为可靠滞后。

## 计算与输出

在同一 168 h 历史窗口上运行两路：

1. raw：168 个原始小时值分别标准化，进行循环互相关。
2. patch：先取 14 个 patch 均值，再沿 14 个时间位置标准化并进行循环互相关。不会将旧滞后除以 12，也不会展平 patch 内的 12 个值作为新的时间序列。

对任一路长度 N 的标准化序列：

`R(d) = sum_k y[k] * x[(k-d) mod N] / N`

FFT 为 `irfft(rfft(y)*conj(rfft(x)), n=N)/N`。绝对值不小于左右邻居才保留，在 `1..N-2` 选绝对值最大位置；并列取较小位置。保留带符号分数。只读取历史，不读取未来 X/Y。

默认 raw 候选为 1..166 h；patch 候选为 1..12 个位移，即 12、24、……、144 h（13 对应 156 h，仅参与邻居比较）。因此这项对比同时改变了聚合平滑、时间分辨率、局部峰邻域和候选位置；不能单独将结果差异归因于噪声减少或宣称更准确。

一般设置中 `patch_lag_hours = patch_lag_steps * patch_stride_hours`。即使是重叠 patch，也按步长换算，不能按宽度换算。

每个数据集单独保存：

- `DATASET_raw_vs_patch.csv.gz`：同一行包含窗口起点、历史/预测时间范围、变量名、raw_lag_hours、raw_score、patch_lag_steps、patch_lag_hours、patch_score 和回退标记。
- `plots/DATASET_raw_vs_patch.png/.svg`：每行一个外生→OT，左 raw，右 patch，横轴均为小时，每根柱对应一个精确小时。不是按 patch 重新分箱旧结果。
- `plots/DATASET_distribution_hours.csv`：计数和占比。
- `summary.json`、`metadata.json`：汇总、边界、搜索范围、数据哈希及参照提交。

图包含全部窗口，回退数另报；均值仅为循环滞后编号的算术平均。168 h 周步长不消除循环歧义，也不保证窗口之间统计独立。

## 验证

```bash
python -m unittest -v test_compare_patch
# 可选：直接核对基础模型源函数，须使用上述固定提交
EXO_BASE_REFERENCE=../exo-base-model python -m unittest -v test_compare_patch
```

7 项测试覆盖官方不同长度数值对照、1 h 退化等价、已知正滞后及负相关、均值对块内排列不变、半开边界/重叠/裁短尾块、常量回退及参数错误。可选测试从参照源代码提取原函数执行，不复制改写一套假参照；没有提供基础仓库时该项明确跳过。

本次实际执行全部 7 项且通过。五个真实数据集各 311 个历史窗口、622 条配对记录，共 3,110 条记录：raw 和 patch 都与官方估计器的对应变量对逐值一致；与基础模型 FFT 函数的滞后完全一致，patch 分数也完全一致，raw 分数最大浮点差异 2.3842e-7。NP/DE/PJM/FR 的 raw 结果还与仓库已有结果逐值一致。见 verification_patch.json。
