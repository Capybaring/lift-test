# NP / DE / PJM / FR 循环 LIFT 滞后估计

沿用用户确认的设置：历史 168 h，未来 24 h，滑动步长 168 h（7 天），只计算外生→OT。每个数据集独立；按精确 1 h 绘图，不使用 patch 单位。CPU 运行，无需 GPU。

## 运行

解压并进入本目录，安装依赖：

```bash
python -m pip install -r requirements.txt
```

本机一次运行四个数据集：

```bash
python lift_weekly.py --data-dir ~/Downloads/dataset --output ./my_results
```

服务器上一次运行四个：

```bash
python lift_weekly.py --data-dir /your/path/dataset --output ./my_results
```

--data-dir 支持 CSV 所在目录，也支持其上一级（自动查找 forecasting/）。文件名为 NP.csv、DE.csv、PJM.csv、FR.csv。

只运行 NP，或任选多个：

```bash
python lift_weekly.py --datasets NP --input /your/path/NP.csv --output ./np_results
python lift_weekly.py --datasets DE PJM FR --data-dir /your/path/dataset --output ./other_results
```

数据集参数不区分大小写；文件名按原始大写名称查找。输出位于 output/数据集名/。同名输出已存在时会停止，需指定新输出路径或加 --overwrite。
--batch-size 只调整内存使用量，不改变窗口参数或计算方法。

仅对已有结果重新绘图：

```bash
python plot_lags.py --dataset NP --results ./my_results/NP
```

## 数据核对和变量定义

四个源文件均为 date,data,cols 三列长表，每个变量 52,416 个时间点，严格每小时递增，无缺失或非有限值。
内外生定义依据用户提供的 DAG 论文表 1、附录 A.1；其短期预测设置为历史 168 步、未来 24 步。

| 数据集 | 外生变量（显示名称） | 内生变量 |
| --- | --- | --- |
| NP | Grid load forecast；Wind power forecast | OT（电价） |
| DE | Wind power forecast；Ampirion zonal load forecast | OT（电价） |
| PJM | System load forecast；Zonal COMED load foecast | OT（电价） |
| FR | Generation forecast；System load forecast | OT（电价） |

代码保留 CSV 列名前导空格及原拼写，包括 PJM 的 foecast。请勿改源文件列名。

## 输出

每个数据集有 311 个窗口、622 条配对结果：

- DATASET_lift_window_lags.csv.gz：窗口时间、lag_hours、带符号 score、abs_score 和两个外生之间的分数排序。
- summary.json：各配对平均滞后、分数摘要及频数。
- metadata.json：输入校验值、参数、时间范围和环境版本。
- plots/DATASET_lag_distribution_hours.png、.svg：每小时分布图。
- plots/DATASET_lag_distribution_counts.csv：每小时计数和占比。
- DATASET_non_circular_checks.csv.gz：仅对已选循环滞后做辅助核对，绝不重新选择滞后、不影响主图。大滞后下 overlap_n 很小，核对分数不宜直接当作可靠性依据。

压缩包 results/ 内已有本机预运行的完整结果，无需为看图重新运行。
window_id 是历史起点索引 0、168、336……，每个数据集从自己的首行开始，跨数据集不意味着相同日历日期。

## 计算口径

两条 168 点历史序列分别减均值，再除以 sqrt(窗口方差+1e-8)。令标准化外生为 x，目标为 y：

R(d) = mean_k[y[k] x[(k-d) mod 168]]

用 irfft(rfft(y)*conj(rfft(x)), n=168)/168 实现，不零填充。
沿用官方 local_max=True，保留绝对分数不小于左右相邻点的局部峰，在 1..166 h 中选绝对值最大者；0、167 仅作邻居比较。并列取最小滞后，全零保留 lag=1、score=0 回退标记。
每个外生单独选峰，保留全部两个外生，不计算外生互相或内生互相。
未来 24 h 只约束样本完整性，不参与相关计算；本程序不训练 LIFT 的预测网络。
每根柱表示选中该小时滞后的窗口占比，而非相关分数。图中平均值是循环滞后编号的算术平均，不代表固定物理延迟。
按周滑动消除相邻历史窗口重叠，但不能消除循环对齐歧义（例如 144 与 -24 是同一循环位移）。

## 来源与验证

LIFT 官方固定提交：79cf8f157bb9a616c9733daf1e9c2cfefdc98c94。
https://github.com/SJTU-DMTai/LIFT/blob/79cf8f157bb9a616c9733daf1e9c2cfefdc98c94/util/lead_estimate.py

upstream/lead_estimate.py 为原始代码；instance_norm.py 为官方归一化函数提取。实际计算限制为 OT 对两个外生，验证时才运行上游完整矩阵作为参照。
对四个数据集的全部 1,244 个采样窗口，所选滞后与带符号分数均与上游实现逐值一致；另核对时间索引、输出行数和分布占比总和。见 verification.json。
四张 PNG 均已检查显示效果。上游快照无根 LICENSE，保留来源，不另行声明其许可。
