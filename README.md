# mini-nasbench-201
NAS-Bench-201 contains a lot of information and is therefore very big.

In my experience I rarely need access to **all** of the information to run an experiment, just a small subset (i.e. just the model accuracies)

This is a simple way to generate a small file which contains only the information you need. Either you can use the minibench file provided directly or you can generate your own with the info you want. 

On `wolfburn`, loading `minibench` took 0.0062 seconds, where NAS-Bench-201 took 61.0529 seconds.

## Example usage

First, copy the `minibench-arch-cell-accs.pd` file into your repository. 

```python
import pandas as pd

minibench = pd.read_pickle('minibench-arch-cell-accs.pd')

## get accuracy for arch 42 on cifar10-val
cifar10_val = minibench.iloc[42]['cifar10-val']
```

You can also `iloc` with a list of indices. The rows correspond to the `NAS-Bench-201` architecture IDs. 
