# data_pipeline/tokenization/base_tokenizer.py

import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Union, Any, Optional


class BaseTokenizer(ABC):
    """
    分词器基类 - 定义通用的分词器接口。
    所有具体的分词器都应继承此类。
    """

    def __init__(self):
        """初始化基础分词器"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._fitted = False

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """返回词汇表大小"""
        pass

    @property
    def is_fitted(self) -> bool:
        """返回分词器是否已训练"""
        return self._fitted

    @abstractmethod
    def fit(self, text: Union[str, List[str]], **kwargs):
        """
        从文本构建词汇表

        Args:
            text: 训练文本，可以是字符串或字符串列表
            **kwargs: 额外的训练参数
        """
        pass

    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """
        将字符串编码为整数列表

        Args:
            text: 要编码的文本
            **kwargs: 额外的编码参数

        Returns:
            编码后的token ID列表
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int], **kwargs) -> str:
        """
        将整数列表解码为字符串

        Args:
            tokens: token ID列表
            **kwargs: 额外的解码参数

        Returns:
            解码后的字符串
        """
        pass

    def _validate_fitted(self):
        """验证分词器是否已训练"""
        if not self._fitted:
            raise ValueError("分词器尚未训练，请先调用 fit() 方法")

    def _validate_text_input(self, text: Union[str, List[str]]) -> str:
        """
        验证并处理输入文本

        Args:
            text: 输入文本

        Returns:
            处理后的字符串

        Raises:
            ValueError: 当输入文本为空时
        """
        if not text:
            raise ValueError("输入文本不能为空")

        if isinstance(text, list):
            combined_text = "".join(text)
        else:
            combined_text = text

        if not combined_text:
            raise ValueError("文本内容不能为空")

        return combined_text

    def _prepare_save_data(self) -> Dict[str, Any]:
        """
        准备要保存的数据（子类应重写此方法添加特定数据）

        Returns:
            要保存的数据字典
        """
        return {
            "fitted": self._fitted,
            "class_name": self.__class__.__name__,
            "version": "1.0",
        }

    def _load_base_data(self, data: Dict[str, Any]):
        """
        加载基础数据（子类应调用此方法）

        Args:
            data: 加载的数据字典
        """
        self._fitted = data.get("fitted", False)

    def save(self, filepath: Optional[Union[str, Path]] = None, format: str = "pickle"):
        """
        将分词器保存到文件

        Args:
            filepath: 保存路径，如果为None则使用类名作为文件名
            format: 保存格式，可选 "pickle" 或 "json"

        Raises:
            ValueError: 当分词器未训练时或格式不支持时
            OSError: 当文件操作失败时
        """
        self._validate_fitted()

        if format not in ["pickle", "json"]:
            raise ValueError("格式必须是 'pickle' 或 'json'")

        # 如果没有提供文件路径，使用类名作为默认文件名
        if filepath is None:
            extension = ".pkl" if format == "pickle" else ".json"
            filepath = f"{self.__class__.__name__}{extension}"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = self._prepare_save_data()

        try:
            if format == "pickle":
                with open(filepath, "wb") as f:
                    pickle.dump(data_to_save, f)
                self.logger.info(f"分词器已保存到 {filepath} (pickle格式)")
            else:  # json format
                # 对于JSON格式，需要确保所有数据都是JSON可序列化的
                json_data = self._convert_to_json_serializable(data_to_save)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"分词器已保存到 {filepath} (JSON格式)")
        except OSError as e:
            self.logger.error(f"保存失败: {e}")
            raise

    def _convert_to_json_serializable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将数据转换为JSON可序列化格式（子类可重写此方法）

        Args:
            data: 原始数据字典

        Returns:
            JSON可序列化的数据字典
        """
        json_data = {}
        for key, value in data.items():
            if isinstance(value, dict) and any(
                isinstance(k, int) for k in value.keys()
            ):
                # 转换整数key为字符串（JSON限制）
                json_data[key] = {str(k): v for k, v in value.items()}
            else:
                json_data[key] = value
        return json_data

    def load(self, filepath: Union[str, Path]):
        """
        从文件加载分词器，自动检测文件格式

        Args:
            filepath: 文件路径

        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件格式不正确时
            OSError: 当文件操作失败时
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 自动检测文件格式
        try:
            # 首先尝试加载为pickle文件
            with open(filepath, "rb") as f:
                data_loaded = pickle.load(f)
            self.logger.info("检测到pickle格式文件")
        except (pickle.UnpicklingError, UnicodeDecodeError):
            # 如果pickle失败，尝试JSON格式
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data_loaded = json.load(f)
                self.logger.info("检测到JSON格式文件")
            except json.JSONDecodeError as e:
                raise ValueError(f"文件格式错误，既不是有效的pickle也不是JSON: {e}")
        except OSError as e:
            self.logger.error(f"加载失败: {e}")
            raise

        # 验证数据完整性
        self._validate_loaded_data(data_loaded)

        # 加载基础数据
        self._load_base_data(data_loaded)

        # 子类加载具体数据
        self._load_specific_data(data_loaded)

        self.logger.info(f"分词器已从 {filepath} 加载")

    def _validate_loaded_data(self, data: Dict[str, Any]):
        """
        验证加载的数据（子类可重写此方法添加特定验证）

        Args:
            data: 加载的数据字典

        Raises:
            ValueError: 当数据格式不正确时
        """
        required_fields = ["fitted", "class_name"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"文件缺少必要字段: {field}")

    @abstractmethod
    def _load_specific_data(self, data: Dict[str, Any]):
        """
        加载特定于子类的数据

        Args:
            data: 加载的数据字典
        """
        pass

    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """
        获取文本统计信息

        Args:
            text: 输入文本

        Returns:
            统计信息字典
        """
        if not text:
            return {}

        unique_chars = set(text)
        encoded = self.encode(text) if self._fitted else []

        stats = {
            "文本长度": len(text),
            "唯一字符数": len(unique_chars),
            "token数量": len(encoded) if encoded else 0,
            "压缩比": round(len(text) / len(encoded), 2) if encoded else 0,
        }

        if self._fitted:
            stats["词汇表大小"] = self.vocab_size

        return stats

    def __len__(self) -> int:
        """返回词汇表大小"""
        return self.vocab_size

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        status = "fitted" if self._fitted else "not fitted"
        vocab_info = (
            f"vocab_size={self.vocab_size}" if self._fitted else "vocab_size=unknown"
        )
        return f"{self.__class__.__name__}({vocab_info}, status={status})"
