import copy
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer, BartConfig, BartForSequenceClassification, \
    AdamW, get_linear_schedule_with_warmup


class DummyModel(nn.Module):
    def __init__(self, num_classes):
        super(DummyModel, self).__init__()
        self.num_classes = num_classes
        self.x = nn.Parameter(torch.tensor([1], dtype=torch.float))

    def forward(self, input_ids: torch.Tensor, **kwargs):
        batch_size = input_ids.shape[0]
        return torch.rand(1, device=self.x.device) + self.x, torch.rand((batch_size, self.num_classes),
                                                                        device=self.x.device)


class GenreDataset(Dataset):

    def __init__(self, sentences: List[str], labels: List[int], tokenizer: BartTokenizer, max_len: int = 128):
        if len(sentences) != len(labels):
            raise RuntimeError("Sentences and labels should have the same number of elements.")
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.max_len = max_len

    def __getitem__(self, index: int):
        inputs = self.tokenizer(self.sentences[index], truncation=True, pad_to_max_length=True, max_length=self.max_len)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long).squeeze(),
            "attention_mask": torch.tensor(mask, dtype=torch.long).squeeze(),
            "labels": torch.tensor(self.labels[index], dtype=torch.long)
        }

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def load_manually(file_dir: str, tokenizer: BartTokenizer, max_len: int) -> "GenreDataset":
        sentences = []
        labels = []
        label_to_index = {'Metal': 0,
                          'Pop': 1,
                          'Jazz': 2,
                          'Rock': 3,
                          'Folk': 4}
        for f_name in os.listdir(file_dir):
            if f_name.endswith('.csv'):
                print(f_name)
                file_path = Path(os.path.join(file_dir, f_name))
                with open(file_path, encoding='utf-8', mode='r') as in_file:
                    for line in in_file:
                        vals = line.strip().split(',')
                        sentences.append(vals[4])
                        labels.append(label_to_index[vals[5]])
        return GenreDataset(sentences, labels, tokenizer, max_len)


def create_optimizer_scheduler(model: nn.Module, scope: Dict[str, Any], num_training_steps: int
                               ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": scope["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=scope["lr"], eps=scope["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=scope["warmup_steps"], num_training_steps=num_training_steps
    )
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 200, )
    return optimizer, scheduler


def train_epoch(model: nn.Module, device: torch.device, optimizer: Optimizer, training_loader: DataLoader) -> Tuple[
    float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_steps = 0

    for batch in tqdm.tqdm(training_loader):
        ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(
            device)
        retval = model(ids, attention_mask=mask, labels=labels, return_dict=True)
        loss = retval['loss']
        _, preds = torch.max(retval['logits'], dim=1)
        correct = (labels == preds).sum().item()

        total_steps += 1
        total_examples += ids.size()[0]
        total_loss += loss.item()
        total_correct += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = total_correct / total_examples
    avg_loss = total_loss / total_steps
    return avg_loss, accuracy


def test_model(model: nn.Module, device: torch.device, data_loader: DataLoader) -> Tuple[List[int], List[int]]:
    model.eval()

    labels_list = []
    preds_list = []

    with torch.no_grad():
        for batch in data_loader:
            ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(
                device)
            ret_dict = model.forward(ids, attention_mask=mask, labels=labels, return_dict=True)
            _, preds = torch.max(ret_dict['logits'], dim=1)

            labels_list.extend(labels.cpu().numpy().tolist())
            preds_list.extend(preds.cpu().numpy().tolist())

    print(labels_list[:20], preds_list[:20])
    return labels_list, preds_list


def valid_epoch(model: nn.Module, device: torch.device, data_loader: DataLoader) -> Dict[str, float]:
    return compute_metrics(*test_model(model, device, data_loader))


def compute_metrics(labels: List[int], predictions: List[int]) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    conf_matrix = get_confusion_matrix(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix
    }


def get_confusion_matrix(labels: List[int], predictions: List[int]):
    return confusion_matrix(labels, predictions)


def create_model(scope: Dict[str, Any]) -> Tuple[BartForSequenceClassification, BartTokenizer]:
    device = scope["devices"][0]
    model_name = scope["model"]
    config = BartConfig.from_pretrained(model_name)
    config.num_labels = 5

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForSequenceClassification.from_pretrained(model_name, config=config) if not scope[
        "dummy"] else DummyModel(5)
    if scope['model_path'] != None:
        print('Loading pretrained model')
        model_checkpoint = torch.load(scope['model_path'])
        model.load_state_dict(model_checkpoint)
    model.train()
    model.to(device)
    return model, tokenizer


def main_custom_training(scope: Dict[str, Any]):
    device = scope["devices"][0]
    model, tokenizer = create_model(scope)

    writer = SummaryWriter()

    train_dataset = GenreDataset.load_manually(Path(args.train_dir), tokenizer, scope["max_len"])
    valid_dataset = GenreDataset.load_manually(Path(args.validation_dir), tokenizer, scope["max_len"])
    test_dataset = GenreDataset.load_manually(Path(args.test_file), tokenizer, scope["max_len"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)

    epoch_iters = len(train_dataset) // (1 * scope["batch_size"])
    warmup_steps = min(2000, int(0.1 * scope["epochs"] * epoch_iters))
    scope['warmup_steps'] = warmup_steps

    train_steps = len(train_dataset) // (1 * scope["batch_size"])
    optimizer, scheduler = create_optimizer_scheduler(model, scope, train_steps)

    best_f1 = 0
    best_epoch = -1
    best_params = copy.deepcopy(model.state_dict())
    patience = 0

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")

        train_loss, train_accuracy = train_epoch(model, device, optimizer, train_loader)
        print(f"\tTrain loss: {train_loss}\n\tTrain accuracy: {100 * train_accuracy}%")
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_accuracy", train_accuracy, epoch)

        result = valid_epoch(model, device, valid_loader)
        print(f"\tValidation: {result}")
        writer.add_scalar("valid_accuracy", result["accuracy"], epoch)
        writer.add_scalar("valid_f1", result["f1"], epoch)

        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_epoch = epoch
            best_params = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        print(f'Current patience {patience}')
        if patience == 2:
            break
        best_params = copy.deepcopy(model.state_dict())
        scheduler.step()

    print(f"Best epoch: {best_epoch} - F1: {best_f1}")
    model.load_state_dict(best_params)
    result = valid_epoch(model, device, test_loader)
    print(f"Test results: {result}")

    torch.save(model.state_dict(), "model.pth")
    print("Model parameters saved to model.pth")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument("--train_dir")
    parser.add_argument('--validation_dir')
    parser.add_argument("--test_file")
    parser.add_argument("--gpu")
    parser.add_argument("--model", default="base")
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--weight_decay", default=0.01)
    args = parser.parse_args()

    scope = vars(args)
    scope['adam_epsilon'] = 1e-8
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu is not None and len(args.gpu) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in args.gpu))
        scope["devices"] = [torch.device("cuda", x) for x in range(len(args.gpu))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        scope["devices"] = [torch.device("cpu")]
    print("Starting on:")
    for x in scope["devices"]:
        print(f"\t{x}")
    print()
    main_custom_training(scope)
