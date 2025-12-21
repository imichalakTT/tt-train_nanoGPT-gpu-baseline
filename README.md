## GPU baseline for nanoGPT training in tt-train on Tenstorrent hardware
This repository was used to compare tt-train's implementation of nanoGPT training accuracy-wise to GPU

- [Original readme](./original_README.md)


### Quickstart

**Setup venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate
# Install torch for your platform with cuda you have from here: https://pytorch.org/get-started/locally/ 
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy transformers datasets tiktoken wandb tqdm
```

**Download shakespeare dataset**
```bash
python data/shakespeare_char/prepare.py
```

**Run training**
The following script runs 5000 steps of nanoGPT training on shakespeare dataset with 
1:1 architecture and configuration:

```bash
python train.py config/train_shakespeare_char.py
```

The same training in tt-train (at the moment: 255c9332f12f62060b60c878ddff2001e0544cc7) would be: 
```bash
./build/sources/examples/nano_gpt/nano_gpt -c ./configs/training_configs/training_shakespeare_nanogpt.yaml
```


### What was done:
1. I changed adamW configuration to the same as used in tt-train and switched off gradients clipping. See [this](./config/train_shakespeare_char.py)
```python
learning_rate = 3e-4
weight_decay = 0.01
decay_lr = False
grad_clip = 0.0
beta2 = 0.999
```

2. I set default type to bfloat16 to make sure weights and gradients are kept in bfloat16
3. I have added adamW implementation that explicitly uses bfloat16 for optimizer state and doesn't use master weights to align to default tt-train behaviour. See [bf16_adamw.py](./bf16_adamw.py)