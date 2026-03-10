import pandas as pd
from typing import Callable
from torch.utils.data import Dataset
from src.data.metadata import MetadataFields, SplitNames


class CodonBertDataset(Dataset):
    """This dataset expects a CSV file with the following required columns:
    - id: Unique identifier for each sequence
    - value: Target value/label for the sequence (configurable)
    - ref_seq: Reference DNA/RNA sequence (configurable, U will be converted to T)
    
    Optional columns:
    - split: Data split indicator ('train', 'val', 'test')
    """
    OPTIONAL_COLUMNS = ['split']
    
    def __init__(
        self,
        data_path,
        tokenizer,
        process_item,
        split_name=SplitNames.ALL,
        value_col='value',
        ref_seq_col='ref_seq',
        **kwargs
    ):
        """
        Initialize the CodonBertDataset.
        
        Args:
            data_path (str): Path to the CSV file containing the dataset.
                           Must contain columns: 'id', value_col, ref_seq_col
            tokenizer: Tokenizer object used to tokenize sequences
            process_item (Callable): Function to process individual sequence items.
                                   Should accept (sequence, tokenizer) and return dict
            split_name (str, optional): Which data split to use. Options:
                                      - 'all': Use entire dataset
                                      - 'train': Use only training split
                                      - 'val': Use only validation split  
                                      - 'test': Use only test split
                                      Defaults to 'all'
            value_col (str, optional): Column name for target values. Defaults to 'value'
            ref_seq_col (str, optional): Column name for sequences. Defaults to 'ref_seq'
            **kwargs: Additional keyword arguments (currently unused)
        
        Raises:
            FileNotFoundError: If data_path does not exist
            KeyError: If required columns are missing from the CSV
        """
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.value_col = value_col
        self.ref_seq_col = ref_seq_col
        
        # Validate required columns exist
        required_columns = ['id', self.value_col, self.ref_seq_col]
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        self.data[self.ref_seq_col] = self.data[self.ref_seq_col].str.replace('U', 'T')
        self.data = self.data.reset_index(drop=True)
        self.tokenizer = tokenizer
        if split_name == SplitNames.TRAIN:
            self.data = self.data[self.data['split'] == 'train']
        elif split_name == SplitNames.VAL:
            self.data = self.data[self.data['split'] == 'val']
        elif split_name == SplitNames.TEST:
            self.data = self.data[self.data['split'] == 'test']

        self.process_item = process_item

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data.iloc[idx][self.ref_seq_col]
        items = self.process_item(sequence, tokenizer=self.tokenizer)
        items[MetadataFields.LABELS] = self.data.iloc[idx][self.value_col]
        items[MetadataFields.ID] = self.data.iloc[idx]['id']
        return items

    def get_train(self, process_item: Callable = None) -> "CodonBertDataset":
        process_item = process_item if process_item is not None else self.process_item
        return CodonBertDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            process_item=process_item,
            value_col=self.value_col,
            ref_seq_col=self.ref_seq_col,
            split_name=SplitNames.TRAIN
        )

    def get_validation(self, process_item: Callable = None) -> "CodonBertDataset":
        process_item = process_item if process_item is not None else self.process_item
        return CodonBertDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            process_item=process_item,
            value_col=self.value_col,
            ref_seq_col=self.ref_seq_col,
            split_name=SplitNames.VAL
        )

    def get_test(self, process_item: Callable = None) -> "CodonBertDataset":
        process_item = process_item if process_item is not None else self.process_item
        return CodonBertDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            process_item=process_item,
            value_col=self.value_col,
            ref_seq_col=self.ref_seq_col,
            split_name=SplitNames.TEST
        )

    def get_predict(self, process_item: Callable = None) -> "CodonBertDataset":
        process_item = process_item if process_item is not None else self.process_item
        return CodonBertDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            process_item=process_item,
            value_col=self.value_col,
            ref_seq_col=self.ref_seq_col,
            split_name=SplitNames.ALL
        )
