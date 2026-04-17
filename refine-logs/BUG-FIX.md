# Fed-HyPCA Bug Tracker

## B1: Jacobian computation OOM on Qwen3.5-4B

**Severity**: Critical
**Status**: Fixed (2026-04-08)

`_compute_constraint_info_for_server` 中 `torch.autograd.grad()` 在 Qwen3.5-4B 的混合注意力架构下创建巨大计算图，导致 GPU OOM（49GB GPU 上第三次显存峰值触发）。

**Fix**: 禁用 Jacobian 计算，改用全零 Jacobian（标量权重聚合）。
**Location**: `src/federated/client.py:360-363`

---

## B2: 零 Jacobian 下 QP 退化为 FedAvg

**Severity**: Critical
**Status**: Fixed (2026-04-16)

B1 的修复将 Jacobian 设为全零，但 `use_jacobian=True`（默认）仍将代码路由到 QP 路径。零 Jacobian 下 `torch.dot(jac, delta)=0`，约束梯度为零，QP 只优化 consensus 目标，结果等价于 FedAvg。标量重加权路径（`use_jacobian=False`）从未被触发。

**Fix**: 在 server 端自动检测零 Jacobian，强制走标量重加权路径。
**Location**: `src/federated/server.py` — 新增 `_any_nonzero_jacobians()` 检测函数

---

## B3: 标量重加权过弱

**Severity**: High
**Status**: Fixed (2026-04-16)

标量路径使用线性重加权 `w_i * (1 + beta * viol_i)`，`beta=1.0` 时违约值 ~0.4 仅产生 ~40% 权重变化，不足以区分 Fed-HyPCA 与 FedAvg。

**Fix**: 改为指数重加权 `w_i * exp(scalar_beta * viol_i)`，新增 `scalar_reweight_beta=5.0` 参数。
**Location**: `src/federated/aggregation.py:167-186`, `configs/default.py`

---

## B4: 近端项过强阻止个性化

**Severity**: High
**Status**: Fixed (2026-04-16)

`rho=0.1` 导致 consensus_dist=0.37（FedAvg 为 10.66），客户端几乎无法偏离全局模型，不同组织的安全策略差异无法体现在模型参数中。

**Fix**: `rho: 0.1 -> 0.01`
**Location**: `configs/default.py:95`, `train_federated.py:70`, `scripts/run_fedhypca.sh`

---

## B5: 对偶变量逐步更新噪声过大

**Severity**: Medium
**Status**: Fixed (2026-04-16)

`batch_size=1` 时每步仅看一个样本，约束值为二值噪声（单样本要么违约要么不违约），对偶变量剧烈震荡无法收敛。

**Fix**: 新增 `dual_update_interval=10`，每 10 步累积约束值取均值后再更新对偶变量。
**Location**: `src/federated/client.py:259-290`, `configs/default.py`

---

## B6: 对偶步长 eta_lambda 过小

**Severity**: Medium
**Status**: Fixed (2026-04-16)

`eta_lambda=0.01` 导致对偶变量增长缓慢，500 步后 lambda 仅 ~1.5，约束项在 Lagrangian 中占比不足。

**Fix**: `eta_lambda: 0.01 -> 0.05`
**Location**: `configs/default.py:98`, `train_federated.py:73`, `scripts/run_fedhypca.sh`

---

## B7: Python stdout 缓冲导致日志不可见

**Severity**: Low
**Status**: Mitigated

Python 默认缓冲 stdout，通过 `tee` 重定向时日志文件长时间为空，无法实时查看训练进度。

**Mitigation**: 使用 `python -u`（unbuffered）运行；或通过 `screen -X hardcopy` 捕获终端输出。
**Location**: `scripts/run_validation.sh` — 已改用 `python -u`

---

## B8: `_compute_constraint_info_for_server` 验证集重算导致每轮耗时翻倍

**Severity**: High
**Status**: Fixed (2026-04-16)

每轮训练结束后，`_compute_constraint_info_for_server` 对整个验证集（500 样本，batch_size=2 → 250 次前向传播）重新计算约束值，耗时 ~4.5 分钟/客户端，而训练本身仅 ~2 分钟。由于 Jacobian 已禁用（全零），这些约束值仅用于标量重加权，精度要求不高。

**Fix**: 复用训练循环中 `accumulated_constraints` 的最终均值作为约束值，跳过验证集重算。
**Location**: `src/federated/client.py:295-314`
