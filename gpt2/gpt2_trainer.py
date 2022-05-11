import torch, os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from datasets import load_dataset

data_path = 'full_data'
data = load_dataset(data_path)

def add_special_tokens():
	""" Returns GPT2 tokenizer after adding separator and padding tokens """
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length = 1024)
	special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
	num_add_toks = tokenizer.add_special_tokens(special_tokens)
	return tokenizer

tokenizer = add_special_tokens()

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer)) # VERY IMPORTANT

class MathQAData(Dataset):  
    def __init__(self, control_code, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.problems = self.tokenizer(control_code['Problem'])

        self.rationales = []
        for item in control_code['Rationale']:
            self.rationales.append(self.tokenizer.encode(item))
        
        self.max_length = max_length
        self.count = len(self.problems['input_ids'])
        
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        example = dict()
        text = self.tokenizer.encode(self.tokenizer.pad_token)*self.max_length
        content = self.problems['input_ids'][idx] + self.tokenizer.encode(self.tokenizer.sep_token) + self.rationales[idx]
        text[:len(content)] = content
        text = torch.tensor(text)
        example['article'] = text
        example['sum_idx'] = len(self.problems['input_ids'][idx])
        return example

dataset = MathQAData(data['train'], tokenizer)

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

device=torch.device("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=-1
    )
loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
global_step = 0
tr_loss, logging_loss = 0.0, 0.0

epochs = 11
loss = 0
epoch_losses = []
epoch_loss = 0
model = model.to(device)
model.train()
model.zero_grad()
optimizer.zero_grad()

for epoch in range(epochs):
    print("Training Epoch {}".format(epoch + 1))
    for idx, batch in enumerate(train_dataloader):
        inputs, labels = batch['article'], batch['article']
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs).logits
        idx = batch['sum_idx'].item()
        shift_logits = logits[..., idx:-1, :].contiguous()
        shift_labels = labels[..., idx+1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss /= 2    # gradient accumulation steps
        epoch_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #
        tr_loss += loss.item()
        if (idx + 1) % 2 == 0: # 2 gradient accumulation steps
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            logging_loss = tr_loss
            print("loss:", loss.item(), end='\n\n')

    avg_epoch_loss = epoch_loss / dataset.__len__()
    epoch_losses.append(avg_epoch_loss)
    print("Epoch {}: Average Loss = {}".format(epoch, avg_epoch_loss))
    epoch_loss = 0
    if epoch % 2 == 0:
        torch.save(model.state_dict(), "models/gpt2_full_epoch_{}.pth".format(epoch + 1))
