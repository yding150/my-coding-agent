"""
job.py - 定时任务实体与五段式 Cron 表达式解析引擎

架构角色：
    为 `coding_agent.cron` 提供定时任务的数据模型 (`CronJob`)、五段式 Unix Cron
    表达式严格校验器 (`validate_cron`) 与时间点匹配引擎 (`cron_matches`)。

解决的核心痛点：
    1. 在大模型注册任务时即刻拦截缺段、越界、逆向区间、非法步长（如 `*/0`）等畸形表达式；
    2. 按标准 Cron 规范处理 Python `weekday()`（周一=0）与 Unix Cron 星期（周日=0）的映射，
       以及“日与星期均非 `*` 时取并集”的语义边界；
    3. 维护 `pending_delivery` 与分钟级 `last_fired` 防重标记，保障重启恢复与同分钟去重。

暴露的核心接口：
    - `CronJob`: 定时任务数据模型（支持 JSON 字典互转）
    - `validate_cron`: 五段式 Cron 表达式合法性校验
    - `cron_matches`: 计算目标 `datetime` 是否命中 Cron 规则
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class CronJob:
    """定时任务数据模型。"""

    id: str  # 任务全局唯一 ID，如 cron_1a2b3c4d
    cron: str  # 5 段式标准 Cron 表达式
    prompt: str  # 触发时交付给 Agent 的提示词内容
    recurring: bool = True  # 是否周期性重复触发（False 为一次性定时任务）
    durable: bool = True  # 是否持久化落盘（False 为仅当前进程会话有效）
    pending_delivery: bool = False  # 任务是否已到期且等待 Agent 确认接收
    last_fired: Optional[str] = None  # 上一次触发的分钟标记（YYYY-MM-DD HH:MM），防同分钟重复触发

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可持久化 JSON 字典。"""
        return {
            "id": self.id,
            "cron": self.cron,
            "prompt": self.prompt,
            "recurring": self.recurring,
            "durable": self.durable,
            "pending_delivery": self.pending_delivery,
            "last_fired": self.last_fired,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CronJob":
        """从磁盘 JSON 字典反序列化恢复任务实体。"""
        return cls(
            id=data["id"],
            cron=data["cron"],
            prompt=data["prompt"],
            recurring=bool(data.get("recurring", True)),
            durable=bool(data.get("durable", True)),
            pending_delivery=bool(data.get("pending_delivery", False)),
            last_fired=data.get("last_fired"),
        )


def _validate_cron_field(field: str, minimum: int, maximum: int) -> Optional[str]:
    """校验单个 Cron 字段的格式与取值范围。"""
    if field == "*":
        return None

    # 步长语法: */N
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) <= 0:
            return f"无效的步长设置: '{field}'"
        return None

    # 离散枚举语法: N,M,K
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), minimum, maximum)
            if err:
                return err
        return None

    # 区间范围语法: N-M
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"无效的区间范围设置: '{field}'"
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            return f"区间起始值大于结束值: '{field}'"
        if start < minimum or end > maximum:
            return f"区间 '{field}' 超出合法取值边界 [{minimum}-{maximum}]"
        return None

    if not field.isdigit():
        return f"无法识别的字段语法: '{field}'"

    val = int(field)
    if val < minimum or val > maximum:
        return f"数值 {val} 超出合法取值边界 [{minimum}-{maximum}]"

    return None


def validate_cron(cron_expr: str) -> Optional[str]:
    """
    严格校验五段式 Cron 表达式。若合法返回 None，若非法返回具体错误原因。

    格式要求：分(0-59) 时(0-23) 日(1-31) 月(1-12) 星期(0-6, 0为周日)。
    """
    if not isinstance(cron_expr, str):
        return "Cron 表达式必须为字符串类型"

    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Cron 表达式必须包含 5 个字段（分钟 小时 日 月 星期），当前包含 {len(fields)} 个"

    field_rules = [
        ("分钟 (minute)", 0, 59),
        ("小时 (hour)", 0, 23),
        ("日期 (day-of-month)", 1, 31),
        ("月份 (month)", 1, 12),
        ("星期 (day-of-week)", 0, 6),
    ]

    for field_val, (name, min_v, max_v) in zip(fields, field_rules):
        err = _validate_cron_field(field_val, min_v, max_v)
        if err:
            return f"{name} 校验失败: {err}"

    return None


def _cron_field_matches(field: str, value: int) -> bool:
    """计算单个字段是否与当前时间数值相匹配。"""
    if field == "*":
        return True

    if field.startswith("*/"):
        step = int(field[2:])
        return (value % step) == 0

    if "," in field:
        return any(_cron_field_matches(part.strip(), value) for part in field.split(","))

    if "-" in field:
        start, end = field.split("-", 1)
        return int(start) <= value <= int(end)

    return value == int(field)


def cron_matches(cron_expr: str, moment: datetime) -> bool:
    """
    判断指定时刻 (`moment`) 是否命中该 Cron 表达式。

    遵循标准 Unix Cron 规范：日与星期若均指定，命中任一即可。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute, hour, day, month, weekday = fields

    # Why: Python datetime.weekday() 为周一=0..周日=6，而标准 Unix Cron 为周日=0..周六=6，
    # 通过 (weekday + 1) % 7 统一对齐到标准 Cron 星期枚举。
    cron_weekday = (moment.weekday() + 1) % 7

    if not (
        _cron_field_matches(minute, moment.minute)
        and _cron_field_matches(hour, moment.hour)
        and _cron_field_matches(month, moment.month)
    ):
        return False

    day_matches = _cron_field_matches(day, moment.day)
    weekday_matches = _cron_field_matches(weekday, cron_weekday)

    if day == "*" and weekday == "*":
        return True
    if day == "*":
        return weekday_matches
    if weekday == "*":
        return day_matches

    return day_matches or weekday_matches
