import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

from datasets import load_dataset
data_path = '/scratch/sr5796/ML_Project/full_data'
data = load_dataset(data_path)
print("Loaded data set")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

print("Created tokenizer and model instance")

class MathQAData(Dataset):  
    def __init__(self, control_code, max_length=1024):

        self.problems = tokenizer(control_code['Problem'], max_length = max_length, padding = 'max_length', truncation = True, return_tensors = "pt")
        self.rationales = tokenizer(control_code['Rationale'], max_length = max_length, padding = 'max_length', truncation = True, return_tensors = "pt")
        self.count = len(self.problems['input_ids'])
        
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        example = dict()
        example['input'] = dict()
        example['output'] = dict()
        example['input']['input_ids'] = self.problems['input_ids'][idx]
        example['input']['attention_mask'] = self.problems['attention_mask'][idx]
        example['output']['input_ids'] = self.rationales['input_ids'][idx]
        example['output']['attention_mask'] = self.rationales['attention_mask'][idx]
        return example
    
dataset = MathQAData(data['train'])

print("data set ready")

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print("data loader ready")

device=torch.device("cuda")
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=-1
    )
model.train()
model = model.to(device)
print("model onto device")

epochs = 50
loss = 0
print("here..")
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    print("Training Epoch {}".format(epoch + 1))
    for idx, entry in enumerate(train_dataloader):
        attention_mask = entry['input']['attention_mask'].to(device)
        input_ids = entry['input']['input_ids'].to(device)
        lm_labels = entry['output']['input_ids'].to(device)
        decoder_attention_mask = entry['output']['attention_mask'].to(device) 
        output = model(input_ids = input_ids, labels = lm_labels, attention_mask = attention_mask, decoder_attention_mask = decoder_attention_mask)
        loss = output[0]
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        num_batches += 1
        if (idx + 1) % 100 == 0:
            print("Epoch {}: {} examples processed. Current Loss = {}".format(epoch + 1, idx + 1, epoch_loss / (idx + 1)))

    print("Epoch {}: Running Loss = {}".format(epoch + 1, epoch_loss / num_batches))
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), "/scratch/sr5796/ML_Project/t5_base/t5_base_epoch_{}.pth".format(epoch + 1))
print("finish...")
