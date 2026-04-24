# 📘 SYSSPEC 规范编写指南：如何为生成式系统编写完美、完整、结构清晰的 Spec

> **适用对象**：操作系统/文件系统初学者、LLM 辅助开发实践者、希望从“写代码”转向“写设计”的开发者  
> **核心依据**：基于论文 *Sharpen the Spec, Cut the Code: A Case for Generative File System with SYSSPEC* 提炼的规范工程方法论

---

## 🎯 一、 为什么需要写“完美 Spec”？

传统自然语言 Prompt 存在**语义模糊、缺乏全局约束、并发逻辑难以表达**等致命缺陷，导致 LLM 生成复杂系统（如文件系统）时频繁出现接口不匹配、并发死锁、边界条件遗漏等问题。

SYSSPEC 的核心理念是：**用受形式化方法启发的结构化规范，替代模糊的自然语言提示**。一份完美的 Spec 不是代码的草稿，而是系统的**无歧义蓝图**。它要求开发者将精力从“逐行实现”转移到“严谨设计”，从而让 LLM 能够稳定、正确、可演进地生成底层 C 代码。

---

## 🧱 二、 完美 Spec 的三大核心支柱

SYSSPEC 规范由三个正交维度构成，缺一不可：

| 维度 | 作用 | 形式化思想映射 | 解决的核心挑战 |
|:---|:---|:---|:---|
| **1. 功能规范 (Functionality)** | 定义模块“做什么”、状态如何流转 | Hoare 逻辑 (`{P} C {Q}`)、不变量 | 语义模糊、逻辑遗漏 |
| **2. 模块规范 (Modularity)** | 定义模块“如何与其他组件协作” | Rely-Guarantee 契约 | 接口不匹配、依赖爆炸 |
| **3. 并发规范 (Concurrency)** | 定义模块“如何安全地并发执行” | 锁协议、时序约束 | 竞态条件、死锁、LLM 并发幻觉 |

> ✅ **关键设计原则**：功能与并发**必须解耦**。LLM 无法在单一 Prompt 中同时处理复杂业务逻辑与细粒度锁协议。SYSSPEC 采用“两阶段生成”：先写功能，再注入并发。

---

## 📝 三、 手把手编写指南（含标准模板）

### 🔹 1. 功能规范 (Functionality Specification)

功能规范描述模块的状态转换契约。按复杂度分为三级，新手建议从 Level 1 开始：

| 组件 | 编写要求 | 示例 |
|:---|:---|:---|
| **Pre-condition** | 明确输入合法性、调用前系统状态 | `path: NULL-terminated string array`<br>`name: valid string` |
| **Post-condition** | 必须覆盖所有分支（成功/失败），明确返回值与状态变化 | `Case 1: New inode created, Entry inserted, Return 0`<br>`Case 2: Traversal/insertion failure, Return -1` |
| **Invariant** | 全局/跨函数必须始终成立的性质 | `root_inum always exists`<br>`any modification of an inode must occur while holding the corresponding lock` |
| **System Algorithm / Intent** | 高级逻辑指引。Level 2 用 Intent（自然语言目标），Level 3 用 Algorithm（分步策略+性能提示） | `Intent: "successful traversal and insertion"`<br>`Algorithm: (1) traverse common path, (2) lock coupling, (3) checks & ops` |

💡 **写作技巧**：
- 使用**数学化纪律的自然语言**，避免 `if necessary`、`usually` 等模糊词。改用确定性表达：`file size = max(old_size, offset + len)`。
- Post-condition 必须穷举所有返回值路径，否则 LLM 会生成未定义行为。

---

### 🔹 2. 模块规范 (Modularity Specification)

模块规范解决“组件组合”难题。SYSSPEC 将传统的并发线程验证中的 `Rely-Guarantee` 改造为**模块级接口契约**。

| 组件 | 编写要求 | 示例 |
|:---|:---|:---|
| **[RELY]** | 显式声明依赖的其他模块提供的结构体、函数、锁原语。LLM 仅依赖此列表，不猜测内部实现。 | `struct inode{...};`<br>`void lock(struct inode*);`<br>`struct inode* locate(struct inode*, char* path[]);` |
| **[GUARANTEE]** | 声明本模块对外导出的 API 签名及行为承诺。下游模块将基于此构建自己的 RELY。 | `int atomfs_ins(char*[], char*, int, unsigned, unsigned);` |

💡 **写作技巧**：
- **上下文边界控制**：每个模块规范对应生成的代码应 ≤500 LOC（适配当前 LLM 上下文窗口）。
- **依赖闭合性**：`RELY` 中声明的所有函数/结构，必须在其他模块的 `GUARANTEE` 中存在，形成逻辑蕴含链。

---

### 🔹 3. 并发规范 (Concurrency Specification)

并发规范是**独立文档**，专门描述锁的获取、释放与时序。绝不与功能前置/后置条件混写。

| 组件 | 编写要求 | 示例 |
|:---|:---|:---|
| **Locking Pre/Post-condition** | 明确调用前/后当前线程持有的锁状态 | `Pre: cur is locked.`<br>`Post: if target is NULL, no lock owned.` |
| **Locking Algorithm** | 分步骤描述锁的获取顺序、临界区范围、解锁时机 | `1. Call rcu_read_lock()`<br>`2. Acquire spinlock after hash match`<br>`3. Re-check parent before critical section`<br>`4. Unlock before continue/break` |

💡 **写作技巧**：
- 采用**分阶段锁定描述**，明确每个算法步骤的锁状态。
- 使用 `rcu_read_lock()`、`spin_lock()` 等明确原语，LLM 会据此生成对应并发控制代码。
- 规范工具链会自动执行“两阶段生成”：先验功能代码 → 验证通过 → 根据本规范注入锁逻辑。

---

## 📐 四、 标准 Spec 模板（可直接复用）

```markdown
# [模块名称] Specification

## [RELY] (依赖声明)
- struct/enum definitions: ...
- external functions: ...
- lock primitives: ...

## [GUARANTEE] (导出契约)
- API Signature: return_type function_name(params);
- Behavior Promise: ...

## 功能规范 (Functionality)
- Pre-condition:
  - param1: ...
  - system state: ...
- Post-condition:
  - Case 1 (Success): ...
  - Case 2 (Failure/Edge): ...
- Invariant:
  - ...
- System Algorithm / Intent:
  1. ...
  2. ...

## 并发规范 (Concurrency)
- Locking Pre-condition: ...
- Locking Post-condition:
  - Path A: ...
  - Path B: ...
- Locking Algorithm:
  1. ...
  2. ...
  3. ...
```

> 📌 **实战对照**：此模板完全对应论文 `Appendix B: dentry_lookup` 的规范结构。LLM 工具链可直接解析该结构进行分阶段代码生成。

---

## 🌳 五、 规范演进指南：DAG-Structured Spec Patch

文件系统需要持续演进。SYSSPEC 不鼓励直接修改 C 代码，而是通过 **Spec Patch（规范补丁）** 驱动自动重新生成。

### DAG 节点类型
| 节点类型 | 作用 | 编写要点 |
|:---|:---|:---|
| **Leaf Node（叶子）** | 自包含变更，无依赖 | 仅修改单个模块，定义新结构/函数，提供新 Guarantee |
| **Intermediate Node（中间）** | 基于子节点 Guarantee 构建更复杂逻辑 | 依赖 Leaf 的变更，向上提供更高阶 Guarantee |
| **Root Node（根）** | 集成点，替换旧实现 | 提供与原模块**语义等价的 Guarantee**，确保透明替换，作为“提交点” |

### 演进工作流
1. 编写 Patch 的叶子节点规范
2. 工具链自底向上生成依赖模块
3. 逐步合成中间节点
4. 到达根节点时，新特性完整集成，旧实现被安全替换

> ✅ **优势**：避免“牵一发而动全身”的维护噩梦。所有依赖变更在 DAG 中显式管理，LLM 仅重新生成受影响模块。

---

## 🚫 六、 新手避坑指南 & 最佳实践

| 常见错误 | 后果 | SYSSPEC 正确做法 |
|:---|:---|:---|
| 用自然长段落描述逻辑 | LLM 遗漏边界条件或产生歧义 | 拆分为 `Pre/Post/Invariant` 结构化条目 |
| 在功能规范中混写锁逻辑 | 并发死锁、锁顺序错误、LLM 幻觉 | **严格分离**并发规范，采用两阶段生成 |
| 隐式依赖未声明 | 接口不匹配、编译失败 | 所有外部调用必须在 `[RELY]` 显式列出 |
| 后置条件只写成功路径 | 错误处理缺失、状态泄漏 | 必须穷举 `Case 1, Case 2...` 覆盖所有返回分支 |
| 规范过于抽象 | LLM 生成正确但低效的代码（如冒泡排序代替快排） | 补充 `Intent` 或 `System Algorithm` 提供性能/算法指引 |
| 一次性修改全局代码 | 破坏不变量、回归测试失败 | 使用 `DAG Patch` 局部演进，根节点保证语义等价 |

### 🛠️ 工具链协同建议
- 利用 `SpecAssistant`：提交草稿 → 自动格式化 → 调用编译器验证 → 若失败自动抛光规范 → 返回诊断日志。
- 利用 `SpecValidator`：生成的代码需通过规范审查 + 回归测试（如 `xfstests`）双验证。
- **模块化阈值**：单模块生成代码控制在 30K tokens 以内，超出则拆分。

---

## 🏁 七、 总结：从“码农”到“架构师”的范式跃迁

编写完美的 SYSSPEC 规范，本质是**用严谨的设计契约替代试错式编码**。它要求你在落笔前回答三个问题：
1. **它必须做什么？**（功能契约：Pre/Post/Invariant）
2. **它依赖什么？承诺什么？**（模块契约：Rely/Guarantee）
3. **它如何安全并发？**（并发契约：Locking Protocol）

虽然前期设计投入高于传统手写 C 代码，但回报是：
- ✅ **正确性由构造保证**：通过工具链验证，消除 80%+ 维护类提交（论文中 Ext4 82.4% 为 Bug/Maintenance）
- ✅ **无缝演进**：DAG Patch 让特性集成不再引发雪崩式修复
- ✅ **生产力跃升**：论文实测 `Extent` 特性开发耗时降低 3.0×，`Rename` 降低 5.4×

> 📖 **下一步**：下载 SPECFS 仓库，对照 `Appendix B` 的 `dentry_lookup` 完整 Spec，尝试用模板编写一个简单模块（如文件读取路径解析）。让工具链为你生成 C 代码，体验“Spec-Driven Development”的威力。

---
*本教程严格基于论文 §3~§5 及附录内容提炼。规范格式、Rely-Guarantee 契约、两阶段并发生成、DAG Patch 演进机制均与 SYSSPEC 框架设计一致。*