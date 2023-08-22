import torch


is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"


EMBD = 256
HEAD = 4
BLKS = 8
DROP = 0.1
SQNZ = 512 # seq len
VOCB = 10000


class Attention(torch.nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
        self.out_proj = torch.nn.Linear(EMBD, EMBD)
        self.register_buffer(
            "mask", torch.tril(torch.ones(SQNZ, SQNZ).view(1, 1, SQNZ, SQNZ))
        )

    def forward(self, qry, key, val):
        Q_B, Q_S, _ = qry.shape
        K_B, K_S, _ = key.shape
        V_B, V_S, _ = val.shape
        EMBD_HEAD = int(EMBD / HEAD)

        qry = qry.reshape(Q_B, Q_S, HEAD, EMBD_HEAD).transpose(1, 2)
        key = key.reshape(K_B, K_S, HEAD, EMBD_HEAD).transpose(1, 2)
        val = val.reshape(V_B, V_S, HEAD, EMBD_HEAD).transpose(1, 2)

        msk = self.mask[:, :, :Q_S, :Q_S] == 0
        att = qry @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(EMBD_HEAD))
        att = att if self.is_causal == False else att.masked_fill(msk, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        out = (att @ val).transpose(1, 2).reshape(Q_B, Q_S, EMBD)
        return self.out_proj(out)


class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = torch.nn.Linear(EMBD, EMBD * 4)
        self.relu = torch.nn.ReLU()
        self.c_proj = torch.nn.Linear(EMBD * 4, EMBD)
        self.drop = torch.nn.Dropout(DROP)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(EMBD)
        self.qkv = torch.nn.Linear(EMBD, EMBD * 3)
        self.attn = Attention()
        self.ln_2 = torch.nn.LayerNorm(EMBD)
        self.ffww = FeedForward()

    def forward(self, x):
        # (64, 43, 256) in nn.LayerNorm(256) -> normalization should occur in the last dim, which is 256 -> (64, 43, 256)
        # qkv -> (64, 43, 256) in qkv(256, 256*3) -> (64, 43, 768)
        # (64, 43, 768) -> split(256, dim=-1) -> (64, 43, 256)
        q, k, v = self.qkv(self.ln_1(x)).split(EMBD, dim=-1)
        # Attention((64, 43, 256), (64, 43, 256), (64, 43, 256)) -> (64, 43, 256)
        # (64, 43, 256) + (64, 43, 256)
        x = x + self.attn(q, k, v)
        # (64, 43, 256) + (64, 43, 256)
        x = x + self.ffww(self.ln_2(x))
        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(EMBD, EMBD * 3)
        self.qry = torch.nn.Linear(EMBD, EMBD)
        self.key = torch.nn.Linear(EMBD, EMBD)
        self.val = torch.nn.Linear(EMBD, EMBD)
        self.c_att = Attention(is_causal=True)
        self.x_attn = Attention()
        self.ffww = FeedForward()

    def forward(self, src, tgt):
        q, k, v = self.qkv(tgt).split(EMBD, dim=-1)
        tgt = tgt + self.c_att(q, k, v)

        qry = self.qry(tgt)
        key = self.key(src)
        val = self.val(src)

        print(f'qry {qry.shape}, key {key.shape}, val {val.shape}')
        tgt = tgt + self.x_attn(qry, key, val)
        tgt = tgt + self.ffww(tgt)
        return tgt


class T5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embd = torch.nn.Embedding(VOCB, EMBD)
        self.pos_embd = torch.nn.Embedding(SQNZ, EMBD)
        self.enc_blks = torch.nn.ModuleList([EncoderBlock() for _ in range(BLKS)])
        self.dec_blks = torch.nn.ModuleList([DecoderBlock() for _ in range(BLKS)])
        self.vocab = torch.nn.Linear(EMBD, VOCB)

    def forward(self, src, tgt):
        
        # (64, 43) -> (64, 43, 256)
        src = self.tok_embd(src)
        # src = (64, 43, 256) + (43, 256) -> (64, 43, 256) only because the last dim of both tensors is 256
        src = src + self.pos_embd(torch.arange(src.size(1), device=device))
        
        for blk in self.enc_blks:
            src = blk(src) # (64, 43, 256)

        tgt = self.tok_embd(tgt)
        # (64, 51) -> (64, 51, 256)
        tgt = tgt + self.pos_embd(torch.arange(tgt.size(1), device=device))
        for blk in self.dec_blks:
            tgt = blk(src, tgt) # (64, 51, 256)
        
        tgt = self.vocab(tgt)
        # (64, 51, 256) -> (64, 51, 10000)
        return tgt

    def num_params(self):
        gpt_params = sum(p.numel() for p in self.parameters())
        emb_params = self.tok_embd.weight.numel()
        print(f"Total Parameters: {gpt_params} | Embedding: {emb_params}")
        return {"gpt_params": gpt_params, "emb_params": emb_params}

    def translate(self, src, num=5):
        self.eval()
        tgt = torch.tensor([[2]], device=device)
        top_token_ids = None
        for _ in range(num):
            with torch.no_grad():
                out = self(src, tgt)
                
                softmax = torch.nn.Softmax(dim=-1)
                softmax_output = softmax(out)
                top_k = 10
                _, top_token_ids = torch.topk(softmax_output[0, -1, :], top_k)
                print('top_token_ids', top_token_ids)
                
                out = out[:, -1, :]
                nxt = torch.argmax(out, dim=-1, keepdim=True)
            
                if nxt.item() == 3:
                    break
                tgt = torch.cat((tgt, nxt), dim=1)
        self.train()
        return tgt, top_token_ids
