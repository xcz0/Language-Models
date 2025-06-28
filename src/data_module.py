# src/data_module.py
from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.data_pipeline.utils import io


class DataModule(LightningDataModule):
    def __init__(
        self,
        processed_data_path: str,
        batch_size: int = 32,
        rate: Optional[float] = 0.9,
    ):
        super().__init__()
        self.processed_data_path = Path(processed_data_path)
        self.batch_size = batch_size
        self.rate = rate or 0.9

        # 保存超参数，这使得在checkpoint中可以访问它们
        self.save_hyperparameters()

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """
        检查处理后的数据文件是否存在。
        """
        if not self.processed_data_path.exists():
            raise FileNotFoundError(
                f"处理后的数据文件不存在: {self.processed_data_path}\n"
                f"请先运行数据预处理脚本生成处理后的数据文件。"
            )
        print(f"找到处理后的数据文件: {self.processed_data_path}")

    def setup(self, stage: Optional[str] = None):
        """
        从文件加载已处理的数据。
        """
        print("开始加载已处理的数据...")

        # 检查文件是否存在
        if not self.processed_data_path.exists():
            raise FileNotFoundError(
                f"处理后的数据文件不存在: {self.processed_data_path}\n"
                f"请先运行数据预处理脚本生成处理后的数据文件。"
            )

        try:
            # 加载已处理的数据
            data = io.load_processed_data(self.processed_data_path)
            X = data["X"]
            Y = data["Y"]
            self.tokenizer = data["tokenizer"]
        except Exception as e:
            raise RuntimeError(
                f"加载处理后的数据失败: {e}\n"
                f"数据文件可能已损坏，请重新生成处理后的数据文件。"
            )

        print("数据加载完成！")
        print(f"输入张量形状: {X.shape}")
        print(f"目标张量形状: {Y.shape}")
        print(f"词汇表大小: {self.tokenizer.vocab_size}")

        # 创建数据集并进行拆分
        dataset = TensorDataset(X, Y)

        # 按比例拆分训练和验证集
        train_size = int(self.rate * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        print(f"Total sequences: {len(dataset)}")
        print(f"Training sequences: {len(self.train_dataset)}")
        print(f"Validation sequences: {len(self.val_dataset)}")
        print("数据集设置完成！")

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("数据集未初始化，请先调用setup()方法")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("数据集未初始化，请先调用setup()方法")
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )
