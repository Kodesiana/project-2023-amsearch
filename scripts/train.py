import json
import logging
import argparse

from torch.utils.data import DataLoader, Dataset
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    InputExample,
    LoggingHandler,
    SentenceTransformer,
)

class TripletDataset(Dataset):
    def __init__(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            self.dataset = [json.loads(x) for x in f]

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return InputExample(texts=[item["query"], item["positive"], item["negative"]])

    def __len__(self):
        return len(self.dataset)


# https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/training/ms_marco/train_bi-encoder_mnrl.py
def main(args):
    # create data loader
    train_dataset = TripletDataset(args.dataset_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # create model
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length

    # create loss
    train_loss = MultipleNegativesRankingLoss(model=model)

    # train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        use_amp=True,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        # checkpoint_path=args.model_output_path,
        # checkpoint_save_steps=len(train_dataloader),
        optimizer_params={"lr": args.lr},
    )

    # save the model
    model.save(args.output_path)


if __name__ == "__main__":
    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--warmup-steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)

    args = parser.parse_args()
    print(args)

    main(args)
