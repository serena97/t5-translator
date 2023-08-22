import torch
import pandas
import tokenizer


class LangDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.column_names = ["id_eng", "eng", "id_cantonese", "cantonese"]
        self.df = pandas.read_csv(
            "./eng-cantonese.tsv",
            delimiter="\t",
            encoding="utf-8",
            on_bad_lines="skip",
            header=None,
            names=self.column_names,
        )
        self.tk = tokenizer.LangTokenizer()
        self.tk.load()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        contx = self.tk.encode(row["eng"])
        input = [self.tk.sp.bos_id()] + self.tk.encode(row["cantonese"])
        label = (self.tk.encode(row["cantonese"])) + [self.tk.sp.eos_id()]
        return {
            "txt_eng": row["eng"],
            "txt_cantonese": row["cantonese"],
            "contx": torch.tensor(contx),
            "input": torch.tensor(input),
            "label": torch.tensor(label),
        }

    def collate_fn(self, batch):
        contx_pad = torch.nn.utils.rnn.pad_sequence(
            [item["contx"] for item in batch], batch_first=True, padding_value=0
        )
        input_pad = torch.nn.utils.rnn.pad_sequence(
            [item["input"] for item in batch], batch_first=True, padding_value=0
        )
        label_pad = torch.nn.utils.rnn.pad_sequence(
            [item["label"] for item in batch], batch_first=True, padding_value=0
        )

        return {
            "eng": [item["txt_eng"] for item in batch],
            "cantonese": [item["txt_cantonese"] for item in batch],
            "contx": contx_pad,
            "input": input_pad,
            "label": label_pad,
        }


if __name__ == "__main__":
    # ds = Dataset()
    # emma = ds[0]
    # print('emma', emma)
    # 'plain': 'emma'
    # 'input': tensor([ 7, 15, 15,  3])
    # 'label': tensor([15, 15,  3,  1])
    # 'masks': tensor([ 1,  1,  1,  1])
    ds = LangDataset()
    print("len(ds)", len(ds))
    print("ds[362]", ds[362])
