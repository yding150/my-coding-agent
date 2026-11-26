"""
coding_agent.tasks - 双轨计划系统（Phase 0 空占位，计划于里程碑 M2/M4 迁入）。

规划组件：
    - 第一轨（M2）：`TodoManager` 会话级轻量内存待办清单（`todo_write`）；
    - 第二轨（M4）：`TaskStore` 跨会话持久 DAG 任务依赖图（`create_task` 等 6 个工具）。
"""
