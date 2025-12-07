import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed
import os

set_seed(3407)

# -----------------------------------------------------------------------------
# 1. DATASET CLASS
# -----------------------------------------------------------------------------
class WikiDataset(Dataset):
    def __init__(self, split='train', block_size=128, max_examples=None):
        print(f"Loading WikiText-103 ({split})...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        
        if max_examples:
            print(f"Reducing dataset to the first {max_examples} examples for speed...")
            dataset = dataset.select(range(max_examples))

        self.block_size = block_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        print("Tokenizing data (this should take < 30 seconds now)...")
        text_data = "\n".join([x['text'] for x in dataset if len(x['text']) > 0])
        self.tokens = self.tokenizer.encode(text_data)
        print(f"Total tokens in {split} set: {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        dix = torch.tensor(chunk, dtype=torch.long)
        x = dix[:-1]
        y = dix[1:]
        return x, y

# -----------------------------------------------------------------------------
# 2. MAIN TRAINING SCRIPT
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. SETUP DATA
    full_train_dataset = WikiDataset('train', block_size=128, max_examples=5000) 

    # 2. SETUP MODEL
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-micro'  
    model_config.vocab_size = 50257
    model_config.block_size = 128
    model = GPT(model_config)

    # 3. SETUP TRAINER
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 3e-4 
    train_config.max_iters = 2000     
    train_config.batch_size = 64      
    train_config.device = 'cuda'

    trainer = Trainer(train_config, model, full_train_dataset)

    # --- CALLBACK WITH GENERATION ---
    def progress_callback(trainer):
        # 1. Print Loss
        if trainer.iter_num % 100 == 0:
            print(f"iter {trainer.iter_num}: loss {trainer.loss.item():.4f}")
        
        # 2. Generate Text Sample (Every 250 steps)
        if trainer.iter_num % 250 == 0:
            model.eval() # Switch to evaluation mode
            context = "The history of science"
            # Encode prompt
            encoded = full_train_dataset.tokenizer.encode(context)
            x = torch.tensor(encoded, dtype=torch.long)[None,...].to(trainer.device)
            
            # Generate 40 new tokens
            y = model.generate(x, max_new_tokens=40, temperature=1.0, do_sample=True, top_k=10)[0]
            
            # Decode and print
            output = full_train_dataset.tokenizer.decode(y.cpu().numpy())
            print("-" * 50)
            print(f"SAMPLE @ Iter {trainer.iter_num}:")
            print(output)
            print("-" * 50)
            model.train() # Switch back to training mode!

        # 3. Save Checkpoint (Every 500 steps)
        if trainer.iter_num % 500 == 0:
            ckpt_path = os.path.join(os.getcwd(), f'wiki_gpt_step_{trainer.iter_num}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Saved checkpoint: {ckpt_path}")

    trainer.set_callback('on_batch_end', progress_callback)

    print("Starting training with live generation...")
    trainer.run()