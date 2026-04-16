# Research Brief

> **Template for document-based input to `/idea-discovery` or `/research-pipeline`.** Provide detailed context instead of a one-line prompt.

## Problem Statement
当前联邦学习下的 LLM SFT / 对齐工作，大多把重点放在训练效率、通信开销和 PEFT 结构设计上，默认存在一个可共享的统一对齐目标；但真实的跨组织协作并不是只有数据异构，更关键的是组织级政策异构：不同组织对安全边界、合规措辞、免责声明、输出风格、拒答策略、转介策略的要求往往彼此不同。若仍然沿用 FedAvg + LoRA/Adapter 这类“平均更新”范式，最终得到的通常是一个“平均政策模型”，容易导致一部分组织觉得模型过于保守，另一部分组织觉得模型过于激进，因此无法真正满足多组织协作中的政策主权需求。

近两三年的相关顶会工作表明，这个问题不能再被视为简单的非 IID 优化问题，而应被看作“异构政策的表达—聚合—保证”问题：一方面，需要让模型显式表达不同组织的偏好/政策，而不是把它们隐含在本地数据中；另一方面，聚合不应只是平均参数，而应体现鲁棒性、公平性与组织代表性；再进一步，还必须回答“关键约束如何被保证不退化”以及“联邦训练中的安全退化与隐蔽投毒如何被纳入评测与防护”。

因此，我想设计一套面向多组织异构政策的联邦条件化约束对齐框架。该框架不直接追求一个单一全局模型，而是学习“共享能力 + 条件化执行不同政策”的统一模型。相比直接照搬已有方法，我更希望在多篇工作的启发下提出一个融合式增强方案：用文本政策 + 向量政策进行混合表示，用显式约束 + 稳定对偶化训练保证关键合规底线，用可调鲁棒聚合避免严格组织被平均化牺牲，并用本地 policy gateway / guardrail与安全退化/投毒评测形成训练期与推理期的闭环保障。

## Background
- **Field**: NLP / Federated Learning / Trustworthy ML
- **Sub-area**: Federated alignment for LLMs; heterogeneous policy alignment; constrained optimization; robust aggregation
- **Key papers I've read**:  
  - Panacea (NeurIPS 2024)
  - MetaAligner (NeurIPS 2024)
  - One-Shot Safety Alignment via Optimal Dualization (NeurIPS 2024)
  - GRPO: Group Robust Preference Optimization (NeurIPS 2024)
  - Fine-tuning Aligned LMs Compromises Safety (ICLR 2024). 
  - Emerging Safety Attack and Defense in Federated Instruction Tuning (ICLR 2025)
  - KNEXA-FL (AAAI 2026)
- **What I already tried**:  
  - FedAvg + LoRA / Adapter 的联邦 SFT
  - 简单 global-local personalization
  - 固定权重的 client objective aggregation
  - 普通 robust aggregation（如 norm clipping / outlier filtering）思路用于 FL 更新过滤
- **What didn't work**:  
  - 平均聚合会把不同组织的拒答边界、风格和 disclaimer 要求压缩成“平均政策”，导致严格组织的政策被破坏。
  - 简单个性化虽然能提升局部效果，但无法提供可执行、可验证的合规/拒答保证。
  - 固定权重聚合不稳定，平均聚合会违反严格政策，纯 min/max 聚合又过于保守，整体效用下降明显。
  - 经典鲁棒聚合对“alignment poisoning”这类隐蔽安全投毒往往无效，因为恶意更新在参数空间里未必表现为明显离群点。

## Constraints
- Compute: 初级规模算力；优先考虑 2B–13B 模型 + PEFT；可使用 A100 / 5090 量级资源和模拟 federated clients
- Timeline: 10–12 周完成首稿，3 个月内达到第一次投稿版本
- Target venue: NeurIPS 2026 / AAAI 2027
- Data / privacy constraint: 不依赖集中收集原始组织对齐数据；政策示例、偏好数据和安全数据默认保留在本地 client 端

## What I'm Looking For
- [✅] Improvement on existing method: federated policy-conditioned constrained alignment with hybrid policy representation
- [✅] Diagnostic study / analysis paper
- [✅] Other: training-time constraints + adaptive robust aggregation + local policy gateway / guardrail

## Domain Knowledge
我当前的核心方法设想可以叫做 Fed-HyPCA（Federated Hybrid Policy-Conditioned Alignment）。它不是简单借鉴单篇论文，而是对多条研究线索做启发式融合与增强：
第一，政策表示层不只使用 policy vector，也不只使用 policy text，而是引入混合政策表示。其中，policy text 用来表达规则语义、支持未见政策与快速更新；policy vector 用来承载连续控制信号，例如合规严格度、拒答阈值、正式度、保守程度、免责声明强度等。模型中加入一个 policy grounding 模块，把文本政策编码到潜在语义空间，再映射为低维 policy vector，以统一“文本可编辑性”和“向量可控性”。这比单独照搬 Panacea 或 MetaAligner 更合理，因为它同时解决了“政策经常变化”和“训练需要稳定注入”的矛盾。

第二，约束保证层不再把安全/合规仅视为软偏好，而是把每个组织的关键底线写成显式约束集合，例如：哪些类别必须拒答、哪些场景必须附带 disclaimer、隐私泄露分数不得超过阈值、某些高风险请求必须转介而非直接回答等。训练时使用受 one-shot dualization 启发的稳定约束优化，把这些 per-organization constraints 融入联邦对齐目标，而不是依赖不稳定的反复 primal-dual 更新。这里的真正创新不在于“用了 dualization”，而在于把集中式安全约束扩展成组织条件化的联邦约束优化问题。

第三，聚合层不采用单纯的平均更新，也不直接照搬硬 max-min，而是设计可调鲁棒聚合机制。基本思想是：同时优化 average utility 和 worst-organization violation，用一个可调参数在“最差组织保证”和“整体平均效用”之间折中；同时，组织权重可以根据历史 violation、累计约束损失或政策满足率动态调整。这样既继承了 GRPO 关于“严格组织不能被平均牺牲”的思想，又避免纯 worst-case 目标过于保守的问题。换言之，聚合在这里不是纯优化细节，而是一个组织级的“社会选择机制”。

第四，系统与威胁模型层采用“训练期共享能力 + 推理期本地执行”的双层保障结构。联邦训练阶段学习共享能力与 policy-conditioned behavior；推理阶段，每个组织通过 local policy gateway / guardrail 快速落地本地政策，使变化快、差异大的规则不必每次都重训共享模型。同时，将安全退化与隐蔽投毒视为方法必须面对的威胁模型：即使没有恶意，联邦多轮微调也可能削弱安全；存在恶意 client 时，还可能出现 alignment poisoning。为此，框架中可加入轻量 post-hoc repair 机制：当检测到某轮后组织约束满足率明显下降时，服务器触发小规模修复阶段。这样，方法形成了“混合政策表示 + 联邦显式约束 + 可调鲁棒聚合 + 本地执行与后验修复”的完整闭环。

第五，评测设计本身也应体现创新。论文不应只报 average helpfulness / safety，而应把以下指标作为主表核心：

- per-organization constraint satisfaction rate
- worst-organization violation
- refusal consistency
- disclaimer correctness
- policy controllability / policy switching generalization
- safety degradation under multi-round FL
- poisoning robustness and repair effectiveness
这样可以把论文重心明确放在“多组织异构政策是否被可靠满足”上，而不是泛泛地做 PEFT 或 FL 调参。

## Non-Goals
- 不想做“又一个 LoRA / Adapter 聚合小改动”，而缺乏组织级政策建模 insight。
- 不想依赖集中式收集所有组织的原始安全/合规数据或偏好标注。
- 不想把第一篇论文做成超重系统论文，例如完整复刻去中心化联邦架构、复杂匹配调度或全新通信协议。
- 不想把潜变量政策推断、动态长期 policy evolution 作为第一版主贡献；这些更适合做扩展实验或后续工作。
- 不想把论文主线变成“单独的攻击检测论文”；安全退化与投毒更适合作为威胁模型、实验 stress test 与补强模块。

## Existing Results (if any)
目前还没有正式实验结果，但已经有较明确的预期与待验证现象：

- Baseline 1: FedAvg + LoRA
    预计平均效用还可以，但会明显出现跨组织 policy inconsistency，尤其在 strict-policy clients 上 violation rate 偏高。
- Baseline 2: global + local personalization
    预计局部性能有所提升，但在“必须拒答 / 必须带 disclaimer / 必须满足阈值”这类硬约束上仍缺乏可验证保证。
- Baseline 3: average / min / max / adaptive aggregation
    预计 average 会伤害严格组织，纯 min/max 会过于保守，而 adaptive robust aggregation 能在公平性与总体效用之间取得更好折中。
- Ablation 1: text-only vs vector-only vs hybrid policy representation
    预计 text-only 更利于泛化但控制不稳，vector-only 更稳定但对新政策适应差，hybrid 表示应兼顾 controllability 与 generalization。
- Ablation 2: soft penalty vs explicit constraints
    预计显式约束能显著提升 per-org constraint satisfaction 与 refusal consistency。
- Stress test: safety degradation / poisoning
    预计多轮 FL 会引入无意安全退化；传统 robust FL defense 难以可靠发现 alignment poisoning；加入 repair stage 和 local gateway 后，policy violation 和 attack success rate 应有改善。