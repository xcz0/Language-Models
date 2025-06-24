# src/data_module.py

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Optional

from src.data_pipeline.tokenization.char_tokenizer import CharTokenizer
from src.data.data_processor import PoetryDataProcessor


class PoetryDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        batch_size: int,
        context_window: int,
        rate: Optional[float] = 0.9,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.context_window = context_window
        self.rate = rate

        # 保存超参数，这使得在checkpoint中可以访问它们
        self.save_hyperparameters()

        # 初始化数据处理器和分词器
        self.data_processor = PoetryDataProcessor(data_path, tokenizer_path)
        self.tokenizer = CharTokenizer()
        self.encoded_data = None

    def prepare_data(self):
        """
        在单个进程上执行的操作：数据清洗、分词器构建等。
        使用PoetryDataProcessor来处理数据清洗任务。
        """
        print("开始数据预处理...")

        # 使用数据处理器加载和清洗数据
        raw_data = self.data_processor.load_raw_data()
        poetry_contents = self.data_processor.extract_poetry_content(raw_data)

        # 打印数据统计信息
        stats = self.data_processor.get_data_stats(poetry_contents)
        print("数据统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 合并文本并构建分词器
        full_text = self.data_processor.combine_texts(poetry_contents)
        self.data_processor.build_and_save_tokenizer(full_text)

        print("数据预处理完成！")

    def setup(self, stage: Optional[str] = None):
        """
        在所有进程上执行的操作：加载分词器、创建数据集、执行拆分。
        """
        print("开始数据集设置...")

        # 1. 加载分词器
        self.tokenizer = self.data_processor.load_tokenizer()

        # 2. 重新加载和处理数据（如果prepare_data在不同进程中未运行）
        raw_data = self.data_processor.load_raw_data()
        poetry_contents = self.data_processor.extract_poetry_content(raw_data)
        full_text = self.data_processor.combine_texts(poetry_contents)

        # 3. 编码文本
        self.encoded_data = self.data_processor.encode_text(full_text)

        # 4. 创建序列对
        X, Y = self.data_processor.create_sequences(
            self.encoded_data, self.context_window
        )

        # 5. 创建数据集并进行拆分
        dataset = TensorDataset(X, Y)

        # 90% 训练, 10% 验证
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )
