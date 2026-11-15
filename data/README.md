# Data

## Source

The benchmark uses the NASA C-MAPSS turbofan engine degradation simulation
dataset (subsets FD001 to FD004), distributed by the NASA Prognostics Center
of Excellence:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Please cite the dataset authors when you use it: A. Saxena, K. Goebel, D. Simon
and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure
simulation", International Conference on Prognostics and Health Management,
2008, https://doi.org/10.1109/PHM.2008.4711414.

## Raw files

The raw text files are not stored in this repository. To work from them,
download the dataset and place the twelve files in `data/raw/`. The benchmark
run recorded these SHA-256 digests, so you can confirm that your copy is the
same:

| File | SHA-256 |
|---|---|
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `train_FD002.txt` | `dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f` |
| `train_FD003.txt` | `2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2` |
| `train_FD004.txt` | `27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `test_FD002.txt` | `de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02` |
| `test_FD003.txt` | `299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790` |
| `test_FD004.txt` | `1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |
| `RUL_FD002.txt` | `c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b` |
| `RUL_FD003.txt` | `df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb` |
| `RUL_FD004.txt` | `196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1` |

On Windows: `certutil -hashfile train_FD001.txt SHA256`. On macOS or Linux:
`shasum -a 256 train_FD001.txt`.

## Processed arrays

`data/processed/FD00x_w30_cap125.npz` holds the windows that the models were
trained and tested on. They can be used directly, without the raw files.

| Array | Shape | Content |
|---|---|---|
| `Xtr` | (windows, 30, 14) | training windows, float32 |
| `ytr` | (windows,) | RUL label at the last cycle of each training window |
| `Xva` | (windows, 30, 14) | validation windows from held-out engines (15% of engines) |
| `yva` | (windows,) | RUL labels of the validation windows |
| `Xte` | (engines, 30, 14) | final 30-cycle window of each test engine |
| `yte` | (engines,) | true RUL of each test engine |
| `te_ids` | (engines,) | engine identifiers of the test set |

| Subset | Training windows | Validation windows | Test engines |
|---|---|---|---|
| FD001 | 15,332 | 2,399 | 100 |
| FD002 | 39,424 | 6,795 | 259 |
| FD003 | 18,749 | 3,071 | 100 |
| FD004 | 45,594 | 8,434 | 248 |

Preprocessing, as recorded in `configs/experiment_config.json`:

- 14 of the 21 sensor channels are kept: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
- window length 30 cycles, stride 1
- RUL labels are capped at 125 cycles, for training and for testing
- sensors are standardised with training statistics; for FD002 and FD004 this
  is done within each of six operating regimes found by k-means
- engines are split 85/15 into training and validation sets

```python
import numpy as np

d = np.load("data/processed/FD001_w30_cap125.npz")
X_test, y_test = d["Xte"], d["yte"]      # (100, 30, 14), (100,)
```
