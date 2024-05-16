## Tensor Parallel and 2D Parallel

This example shows how to apply tensor-parallelism to your model (here Llama 2 7B) with the `ModelParallelStrategy`, and how it can be combined with FSDP (2D parallelism).
PyTorch 2.3+ and a machine with at least 4 GPUs and 24 GB memory each are required to run this example.

```bash
pip install 'torch>=2.3'
```

Navigate to this example folder and run the training script:

```bash
cd examples/pytorch/tensor_parallel
python train.py
```

You should see an output like this:

TODO

> \[!NOTE\]
> The `ModelParallelStrategy` is experimental and subject to change. Report issues on [GitHub](https://github.com/Lightning-AI/pytorch-lightning/issues).
