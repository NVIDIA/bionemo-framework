from pathlib import Path
from bionemo.core.data.load import load
from bionemo.geneformer.data.block_sampling import MapStyleScDataset, scDataset
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from torch.utils.data import DataLoader
from bionemo.core.data.multi_epoch_dataset import MultiEpochDatasetResampler
import time
import tqdm
import functools
from bionemo.llm.data import collate

from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
def make_dataset():
    data_path: Path = load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl" / "train"

    train_data_path = Path("/home/ubuntu/data/cellxgene_2023-12-15/train")

    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            tokenizer, median_dict = tokenizer, median_dict
        case _:
                raise ValueError("Preprocessing must have failed.")

    dataset = SingleCellDataset(train_data_path, tokenizer=tokenizer, median_dict=median_dict, max_len=2048)
    print("done loading ds")
    return dataset

def get_configs():
    return [
        {
            "block_size": 64,
            "batch_size": 128 * 8,
            "fetch_factor": 8,
            "seed": 42
        }
    ]

def mapstyle_throughput():
    dataset = make_dataset()
    tokenizer = dataset.tokenizer

    configs = get_configs()
    for config in configs:
        factor = config["fetch_factor"] * config["batch_size"]
        extra = len(dataset) % factor
        to_add = factor - extra 
        num_samples = (len(dataset) + to_add) 

        dataset = MultiEpochDatasetResampler(
            dataset, 
            num_samples=num_samples,
            shuffle=False,
        )
        '''
        When we stack the datasets this way, a whole vector is passed into getitem for 
        MultiEpochDatasetResampler
        '''
        dataset = MapStyleScDataset(dataset, **config)

        start = time.time()
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=16,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=tokenizer.token_to_id(GeneTokenizer.pad_token),
                min_length=2048,
                max_length=2048,
            ),
        
        )


        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            if i > 100 * config["fetch_factor"]:
                break
            pass

        end = time.time()
        print(f"MapStyleScDataset: {end - start} seconds")
        print(f"MapStyleScDataset: { 800 * config['batch_size'] / ( end - start)} samples per second")

def iterstyle_throughput():
    dataset = make_dataset()
    tokenizer = dataset.tokenizer

    configs = get_configs()

    for config in configs:
        num_samples = (len(dataset)  - (len(dataset) % (config["batch_size"] * config["block_size"]))) * 2
        dataset = MultiEpochDatasetResampler(
            dataset, 
            num_samples=num_samples,
            shuffle=False,
        )
        # TODO get some intermediate metrics
        dataset = scDataset(dataset, bionemo_permute=False, **config)

        start = time.time()
        dataloader = DataLoader(dataset, batch_size=None, num_workers=16, shuffle=False, 
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=tokenizer.token_to_id(GeneTokenizer.pad_token),
                min_length=2048,
                max_length=2048,
            ),
        )

        # I think this just happens if its not an even multiple of the batch size
        try:
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                if i > 100 * config["fetch_factor"]:
                    break
                pass
        except RuntimeError as e:
            print(e)

        end = time.time()
        print(f"IterStyleDataset: {end - start} seconds")
        print(f"IterStyleDataset: { 800 * config['batch_size']/ ( end - start)} samples per second")

if __name__ == "__main__":
    mapstyle_throughput() 
    iterstyle_throughput()