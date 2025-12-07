import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer

# -----------------------------------------------------------------------------
# 1. Create a Dataset Class for minGPT
# -----------------------------------------------------------------------------
class WikiDataset(Dataset):
    def __init__(self, split='train', block_size=128, max_examples=None):
        """
        Args:
            split: 'train', 'validation', or 'test'
            block_size: length of the context window (T)
            max_examples: limit dataset size for debugging/fast testing
        """
        print(f"Loading WikiText-103 ({split})...")
        # We use wikitext-103-v1, a pre-cleaned slice of Wikipedia
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        
        if max_examples:
            dataset = dataset.select(range(max_examples))

        self.block_size = block_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # We concat all text into one huge string to handle cross-article context
        # (In production, you might want to handle this differently to avoid mixing articles)
        print("Tokenizing data...")
        text_data = "\n".join([x['text'] for x in dataset if len(x['text']) > 0])
        self.tokens = self.tokenizer.encode(text_data)
        print(f"Total tokens in {split} set: {len(self.tokens)}")

    def __len__(self):
        # We return chunks of size block_size
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of data of length block_size + 1
        chunk = self.tokens[idx : idx + self.block_size + 1]
        
        # decode to tensor
        dix = torch.tensor(chunk, dtype=torch.long)
        
        # Inputs (x) are the first block_size tokens
        # Targets (y) are the tokens shifted by 1
        x = dix[:-1]
        y = dix[1:]
        return x, y

# -----------------------------------------------------------------------------
# 2. Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    # --- Configuration ---
    BLOCK_SIZE = 128  # Context length (increase to 1024 if you have a big GPU)
    BATCH_SIZE = 12   # Reduce this if you hit OutOfMemory errors
    MAX_ITERS = 2000  # How many training steps to take
    
    # --- Data Setup ---
    # We limit examples to 5000 docs for speed. Remove 'max_examples' for full training.
    train_dataset = WikiDataset('train', block_size=BLOCK_SIZE, max_examples=5000)
    
    # --- Model Setup ---
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano' # Options: gpt-nano, gpt-micro, gpt-mini, gpt2
    model_config.vocab_size = 50257      # Standard GPT-2 vocab size
    model_config.block_size = BLOCK_SIZE
    model = GPT(model_config)

    # --- Trainer Setup ---
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # standard for small GPTs
    train_config.max_iters = MAX_ITERS
    train_config.batch_size = BATCH_SIZE
    
    trainer = Trainer(train_config, model, train_dataset)
    
    # --- Define a callback to print generation samples periodically ---
    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            context = "The history of science is"
            x = torch.tensor(train_dataset.tokenizer.encode(context), dtype=torch.long)[None,...].to(trainer.device)
            y = model.generate(x, max_new_tokens=30, temperature=1.0, do_sample=True, top_k=10)[0]
            completion = train_dataset.tokenizer.decode(y.cpu().numpy())
            print("------------------------------------------------")
            print(f"SAMPLE @ Iter {trainer.iter_num}: {completion}")
            print("------------------------------------------------")
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # --- Run ---
    print("Starting training...")
    trainer.run()