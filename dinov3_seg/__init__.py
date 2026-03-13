"""dinov3_seg 包的公共导出。

当前子模块采用集中式配置入口，外部脚本通常只需要导入
``get_config`` 即可拿到完整实验配置。

这个 ``__init__`` 保持得很轻，只做两件事：
1. 给包本身补一段总览说明，便于 IDE 悬停时快速理解用途。
2. 导出最常用的配置入口，减少外部脚本的导入层级。
"""

# 统一从包顶层暴露配置工厂函数，方便写成 from dinov3_seg import get_config。
from .config import get_config
