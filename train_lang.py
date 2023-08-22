import torch
import t5
import dataset
import tokenizer
import random


is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"
print('device', device)

random.seed(42)
torch.manual_seed(42)
myT5 = t5.T5().to(device)
myT5.num_params()


tk = (tokenizer.LangTokenizer()).load()
ds = dataset.LangDataset()
dl = torch.utils.data.DataLoader(
    ds, batch_size=64, shuffle=True, collate_fn=ds.collate_fn
)
opt = torch.optim.Adam(myT5.parameters(), lr=0.0001)


for epoch in range(1):
    org = "I"
    src = torch.tensor([tk.encode(org)]).to(device)
    trs = myT5.translate(src)
    print(f"{org} - {tk.decode(trs.tolist()[0])}")

    for idx, batch in enumerate(dl):
        c = batch["contx"].to(device)
        x = batch["input"].to(device)
        y = batch["label"].to(device)
        print('start')
        tk = tokenizer.LangTokenizer()
        print('c', tk.decode((batch["contx"][0].tolist())))
        print('x', tk.decode((batch["input"][0].tolist())))
        p = myT5(c, x)
        # (64, 51, 10000)
        print('FINAL p.shape', p.shape)
        p = p.view(-1, p.size(-1))
        print('FINAL p.shape transformed, 64*51, 10000', p.shape)
        print('y before transformed', y.shape)
        y = y.view(-1)
        print('FINAL p.shape transformed 64*51', y.shape)
        l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)
        if idx % 1000 == 0:
            print(f"Loss: {l.item():.4f}")
        # if idx % 5000 == 0:
        #     torch.save(myT5.state_dict(), f"weights_{epoch}_{idx}.pt")
        l.backward()
        opt.step()
        opt.zero_grad()
        break
