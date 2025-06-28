# data_pipeline/utils/io.py

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """从JSON文件加载数据。"""
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"成功从 {file_path} 加载 {len(data)} 条原始数据。")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}")
        raise


def save_processed_data(file_path: Path, data: Dict[str, Any]):
    """将处理好的数据保存到Pickle文件。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"处理好的数据已保存到: {file_path}")
    except Exception as e:
        logger.error(f"保存处理数据失败: {e}")
        raise


def load_processed_data(file_path: Path) -> Dict[str, Any]:
    """从Pickle文件加载处理好的数据。"""
    if not file_path.exists():
        raise FileNotFoundError(f"处理数据文件不存在: {file_path}")
    try:
        with open(file_path, "rb") as f:
            processed_data = pickle.load(f)
        logger.info(f"成功从 {file_path} 加载处理好的数据。")
        return processed_data
    except Exception as e:
        logger.error(f"加载处理数据失败: {e}")
        raise
