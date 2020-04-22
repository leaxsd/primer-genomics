# Deep Learning in Genomics Primer (PyTorch)

A reproduction of the primer tutorial in deep learning genomics proposed in [Nature Genetics (2018)](https://doi.org/10.1038/s41588-018-0295-5)  in Pytorch. 

For details, please refer to the original notebook [here](https://colab.research.google.com/drive/1UR6fOxF_lgg1Qo3anyPyZgpz-jC7gtZc#scrollTo=_5D80hwMhnaf).

## Files Structure

```basic
src/
├── data
│   ├── accuracy.npy
│   ├── models
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   ├── precision.npy
│   ├── recall.npy
│   ├── test_label.npy
│   ├── test_pred.npy
│   ├── train_loss.npy
│   └── val_loss.npy
├── metrics.py
├── model.py
├── results.ipynb
├── run.py
├── train.py
└── tutorial.py

```

## Results

check here : [results.ipynb](src/results.ipynb)