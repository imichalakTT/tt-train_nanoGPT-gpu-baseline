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

#### Run training

tt-train commit at the moment of writing this README: 255c9332f12f62060b60c878ddff2001e0544cc7

**nanoGPT**
The following script runs 5000 steps of nanoGPT training on shakespeare dataset with 
1:1 architecture and configuration:

```bash
python train.py config/tt-train-aligned/train_shakespeare_char_nanogpt.py
```

The same training in tt-train would be: 
```bash
./build/sources/examples/nano_gpt/nano_gpt -c ./configs/training_configs/training_shakespeare_nanogpt.yaml
```

**gpt2s**

```bash
python train.py config/tt-train-aligned/train_shakespeare_bpe_gpt2s.py
```

The same training in tt-train would be

```bash
./build/sources/examples/nano_gpt/nano_gpt -c ./configs/training_configs/training_shakespeare_gpt2.yaml
```

### What was done:
1. I have aligned adamw configuration and switched off gradients clipping. See [nanogpt config](./config/tt-train-aligned/train_shakespeare_char_nanogpt.py) and [gpt2s config](./config/tt-train-aligned/train_shakespeare_bpe_gpt2s.py)
2. I set default type to bfloat16 to make sure weights and gradients are kept in bfloat16
3. I have added adamW implementation that explicitly uses bfloat16 for optimizer state and doesn't use master weights to align to default tt-train behaviour. See [bf16_adamw.py](./bf16_adamw.py)


### Results
- nanoGPT training: char tokenizer / **~1.2-1.25 @ 5k steps** / RTX5080
```
iter 4900: loss 1.2428, time 16.37ms, mfu 22.73%
iter 4910: loss 1.2302, time 16.45ms, mfu 22.72%
iter 4920: loss 1.2260, time 16.40ms, mfu 22.72%
iter 4930: loss 1.2403, time 16.38ms, mfu 22.73%
iter 4940: loss 1.2118, time 16.39ms, mfu 22.73%
iter 4950: loss 1.2350, time 16.38ms, mfu 22.73%
iter 4960: loss 1.2474, time 16.39ms, mfu 22.73%
iter 4970: loss 1.2090, time 16.41ms, mfu 22.73%
iter 4980: loss 1.2198, time 16.40ms, mfu 22.73%
iter 4990: loss 1.2353, time 16.43ms, mfu 22.72%
step 5000: train loss 1.1250, val loss 1.4531
```

- GPT2s training: BPE tokenizer / **~0.95-1.3 @ 5k steps** / RTX5080
```
iter 4990: loss 1.1676, time 46.87ms, mfu 23.94%
iter 4991: loss 1.1090, time 46.86ms, mfu 23.95%
iter 4992: loss 1.3205, time 46.91ms, mfu 23.94%
iter 4993: loss 1.0081, time 46.98ms, mfu 23.94%
iter 4994: loss 1.1276, time 46.87ms, mfu 23.94%
iter 4995: loss 1.2292, time 46.92ms, mfu 23.94%
iter 4996: loss 0.9673, time 46.97ms, mfu 23.94%
iter 4997: loss 1.1163, time 47.22ms, mfu 23.92%
iter 4998: loss 1.0438, time 46.98ms, mfu 23.92%
iter 4999: loss 1.1242, time 46.95ms, mfu 23.92%
step 5000: train loss 0.7227, val loss 5.9062
```