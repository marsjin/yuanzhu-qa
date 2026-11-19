"""域外话题门控：无书内实体、无读书主题词，且命中域外话题词时判为超出语料。

覆盖率门控按查询词命中比例计算，书内高频常见词（地名、时间词）会把
域外问题抬过阈值。这里加一层更窄的判定：只有当问题既没有书内实体、
也没有读书类主题词、又出现明确的域外话题词时才拒答。
"""

from __future__ import annotations

import re

OFF_TOPIC_RE = re.compile(
    r"天气|气温|下雨|空气|雾霾|股票|基金|汇率|房价|彩票|"
    r"编程|代码|python|算法|报错|"
    r"菜谱|红烧|做法|减肥|健身|旅游|挂号|看病|"
    r"新闻|时政|翻译|作业|考试|论文|"
    r"推荐.*电影|电影推荐",
    re.IGNORECASE,
)

BOOK_TOPIC_RE = re.compile(
    r"小说|这本书|书中|原著|原文|情节|主线|人物|角色|结局|作者|章节|故事|剧情|主题|文化属性"
)


def is_out_of_scope(question: str, entities: list[str]) -> bool:
    """判断问题是否明显超出单书客服范围。"""
    if entities or BOOK_TOPIC_RE.search(question):
        return False
    return bool(OFF_TOPIC_RE.search(question))
