# data_pipeline/processing/sequencing.py

import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def create_sequences(
    encoded_data: torch.Tensor, context_window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从编码数据创建训练序列对 (输入, 目标)。

    Args:
        encoded_data: 编码后的数据张量 (1D)。
        context_window: 上下文窗口大小。

    Returns:
        (输入张量, 目标张量) 的元组。
    """
    if len(encoded_data) <= context_window:
        raise ValueError(
            f"数据长度 {len(encoded_data)} 必须大于上下文窗口 {context_window}"
        )

    num_sequences = len(encoded_data) - context_window
    X = torch.stack(
        [encoded_data[i : i + context_window] for i in range(num_sequences)]
    )
    Y = torch.stack(
        [encoded_data[i + 1 : i + context_window + 1] for i in range(num_sequences)]
    )

    logger.info(
        f"创建了 {num_sequences} 个序列对。输入形状: {X.shape}, 目标形状: {Y.shape}"
    )
    return X, Y
