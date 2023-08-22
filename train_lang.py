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

torch.cuda.empty_cache()
for epoch in range(2):
    org = "I"
    src = torch.tensor([tk.encode(org)]).to(device)
    trs, top_token_ids = myT5.translate(src)
    print(f"{org} - {tk.decode(trs.tolist()[0])}")
    for tokens in top_token_ids:
        print(f"most probable tokens - {tk.decode(tokens.tolist())}")

    for idx, batch in enumerate(dl):
        c = batch["contx"].to(device)
        x = batch["input"].to(device)
        y = batch["label"].to(device)
        tk = tokenizer.LangTokenizer()
        p = myT5(c, x)
        torch.cuda.empty_cache()
        # (64, 51, 10000)
        p = p.view(-1, p.size(-1))
        y = y.view(-1)
        l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)
        if idx % 1000 == 0:
            print(f"Loss: {l.item():.4f}")
        # if idx % 5000 == 0:
        #     torch.save(myT5.state_dict(), f"weights_{epoch}_{idx}.pt")
        l.backward()
        opt.step()
        opt.zero_grad()    
    