---
name: code-review
description: 执行严格的代码审查，包含安全性、正确性、性能与可维护性深度分析。当用户要求代码走查、排查 bug 或审计代码库时使用。
---

# 代码审查技能规范 (Code Review Skill)

你现在拥有资深的代码审查与审计专业能力。在对代码进行审查时，请严格遵照以下结构化审查指南：

## 审查核对清单 (Review Checklist)

### 1. 安全性 (Security - 极重要)

重点检查以下高危隐患：
- [ ] **注入漏洞**：SQL 注入、命令注入、XSS、模板注入
- [ ] **认证与凭据**：硬编码的密码、密钥、弱身份认证逻辑
- [ ] **权限与访问控制**：缺失鉴权、IDOR（越权访问漏洞）
- [ ] **敏感信息泄露**：日志或错误提示中输出敏感数据/密钥
- [ ] **加密与安全哈希**：弱加密算法、不当的密钥管理
- [ ] **第三方依赖安全**：已知 CVE 漏洞组件（通过 `npm audit`, `pip-audit` 扫描）

```bash
# 常用快速安全扫描命令
npm audit                    # Node.js 依赖审计
pip-audit                    # Python 依赖审计
cargo audit                  # Rust 依赖审计
grep -r "password\|secret\|api_key" --include="*.py" --include="*.js"
```

### 2. 正确性 (Correctness)

重点检查逻辑漏洞与边界条件：
- [ ] **逻辑错误**：差一错误 (Off-by-one)、空指针/None 处理、特殊边界未兜底
- [ ] **并发与竞态条件**：多线程/协程未加锁同步访问共享资源
- [ ] **资源泄露**：文件句柄、网络连接、内存未及时释放或关闭
- [ ] **异常处理**：静默吞掉异常 (Swallowed exceptions)、缺失容错路径
- [ ] **类型安全**：不当的隐式类型转换、过度滥用 Any 类型

### 3. 性能表现 (Performance)

重点检查性能瓶颈与效率瓶颈：
- [ ] **N+1 查询问题**：在循环体内部频繁发起数据库查询或网络请求
- [ ] **内存浪费**：一次性加载超大文件/数据集到内存、无用的全局常驻对象
- [ ] **阻塞操作**：在异步代码（asyncio）中执行阻塞式同步 I/O
- [ ] **算法低效**：可以用 O(n) 完成的操作却写成了 O(n^2) 嵌套循环
- [ ] **缺失缓存机制**：频繁重复计算高耗时、高昂的结果

### 4. 可维护性与代码整洁度 (Maintainability)

重点检查代码规范与重构机会：
- [ ] **命名规范**：函数、变量名是否清晰、达意、一致
- [ ] **复杂度控制**：单函数超过 50 行、嵌套层级超过 3 层
- [ ] **代码重复**：大段复制粘贴的冗余逻辑（DRY 原则）
- [ ] **僵尸代码**：无用的 import 引用、永远无法进入的死分支
- [ ] **注释与文档**：过时、误导性注释，或关键晦涩业务逻辑缺少必要解释

### 5. 测试充分性 (Testing)

重点检查自动化测试覆盖：
- [ ] **测试覆盖率**：核心业务主路径是否有单元测试覆盖
- [ ] **边界测试**：是否覆盖了 None、空列表、超长输入、极值等边界用例
- [ ] **外部依赖隔离**：网络、数据库、外部 API 是否使用了合理 Mock
- [ ] **有效断言**：断言是否具体、有意义，而非无脑 `assert True`

## 审查报告输出标准模板 (Review Output Format)

当完成审查后，请按照以下 Markdown 格式结构化输出结果：

```markdown
## 代码审查报告: [文件或组件名称]

### 概要结论 (Summary)
[用 1-2 句话客观评价当前代码质量与整体状态]

### 严重缺陷与安全隐患 (Critical Issues)
1. **[缺陷名称]** (第 X 行): [缺陷详细描述]
   - 危害影响: [可能导致的后果，如数据损坏、命令执行、程序崩溃]
   - 修复建议: [具体的修复代码或方案]

### 优化建议 (Improvements)
1. **[改进点]** (第 X 行): [可提升性能、整洁度或可读性的建议]

### 亮点与优秀实践 (Positive Notes)
- [代码中做得好、值得肯定的架构或实现模式]

### 综合评审结论 (Verdict)
[ ] 准予合并 (Ready to merge)
[ ] 需小幅修改 (Needs minor changes)
[ ] 需重大重构 (Needs major revision)
```

## 常见典型反模式与修改示范 (Common Patterns to Flag)

### Python
```python
# 错误：SQL 注入风险
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
# 正确：参数化查询
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# 错误：命令注入风险
os.system(f"ls {user_input}")
# 正确：参数列表隔离
subprocess.run(["ls", user_input], check=True)

# 错误：可变默认参数陷阱（所有调用共享同一个列表实例）
def append(item, lst=[]):
# 正确：使用 None 作为默认值
def append(item, lst=None):
    lst = lst or []
```

### JavaScript / TypeScript
```javascript
// 错误：原型链污染风险
Object.assign(target, userInput)
// 正确：过滤输入字段
Object.assign(target, sanitize(userInput))

// 错误：使用 eval 执行用户传入代码
eval(userCode)
// 正确：严禁使用 eval 执行不受信任的代码

// 错误：回调地狱 (Callback Hell)
getData(x => process(x, y => save(y, z => done(z))))
// 正确：现代 async/await 链
const data = await getData();
const processed = await process(data);
await save(processed);
```

## 审查工作流 (Review Workflow)

1. **理解上下文**：明确被审查代码的业务目标和所属模块定位；
2. **静态检索**：先使用 grep/Select-String 筛查敏感关键词与 TODO 标记；
3. **自顶向下阅读**：从核心入口逻辑读起，梳理数据流和控制流；
4. **对照 Checklist 逐项核验**：分别按安全性、正确性、性能、整洁度过筛；
5. **输出结构化报告**：指出具体行号，指出原因，并给出可直接复制的修改示范。
