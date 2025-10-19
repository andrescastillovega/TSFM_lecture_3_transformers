# Distributed training
### By Matej S

## 1. Why we need to distribute
* Memory
* FLOPs
* TPUs scale better than GPUs

### Comunication primitives
* Reduce-scatter
* All-gather
    * Data sharded across devices, we need to gather them back
    * Organized into a ring
    * Used before MoEs
* All-reduce
* All2All

### A little primer on transformers
* Stack odf decoder layers
* Attention grows quadratical in seq length
* num_flops_per_token - 6 * num_params

## 2. How to scale
* The naive way
    * Data parallelism (DP)
    * replicate model across devices
    * shard data across devices
* Less naive way to scale
    * FSDP
    * THis is also data parallelism
    * Review the *Some nice tricks* slide
        * pre-fetching
        * bucketing
        * as activations grow, we can throw away the older weights to save memory
* Tensor parallelism
    > ## To Review!
    > Collective matmuls (To review!)
* Context parallelism
    * Mid-training:
        * Introduce high quality samples
        * Increase seq lenght from 8K to 32K (given the increase in quality)
    * Attention can be distributed in several GPUs
        * Ring attention
        * Local shards of Q, K, Vs
        * Load-balancing with causal masking
            * GPU with first rows will be more free, so compute needs to be assigned propperly
* Expert parallelism
    * Used in MoE
    * Naively just shard each expert on separate GPU
    > ## To review!
    > * Review Megatron paper from NVIDIA!
    > * Deepspeed
* Pipeline Parallelism
    * split layers across devices
    * Quite useful but scheduling could be complicated
> ## Tips and tricks
> * Gradient accumulation
> * activation checkpointing
> * flash-attention
> * `PYTHORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

## 3. Practical session
> ## To review!
> `torchtitan` github repo
