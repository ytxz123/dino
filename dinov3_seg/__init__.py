"""dinov3_seg 包的公共导出。

当前子模块采用集中式配置入口，外部脚本通常只需要导入
``get_config`` 即可拿到完整实验配置。
"""

from .config import get_config
