## Implementation Notes

### Activation checkpointing and Fully Sharded Data Parallel

When combining gradient checkpointing with FSDP, the ordering of the two wrapping operations matters. Reference implementations are inconsistent on this point:

- `llama-recipes/src/llama_recipes/finetuning.py`:
  applies FSDP first (line 335), then gradient checkpointing (lines 365–367)

- <https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_fsdp.py>:
  applies gradient checkpointing first (lines 302–304), then FSDP (line 306)

- <https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_fsdp_with_ckpt.py>:
  applies gradient checkpointing first, then FSDP — i.e. `FSDP(checkpointing(module))`
