Put trained weights here (`*.pt`). Every file is listed under **Settings → Model** in the UI.

- `snn_checkpoint.pt`, `cnn_checkpoint.pt`, `lstm_checkpoint.pt` come from `train.py` in snn-keyword-spotter (`outputs/`).
- The model type and size are read from the weights; an optional `<name>.json` sets the label, test accuracy and note shown in the UI.
- `*_personal.pt` files are created by voice calibration and are specific to the person who recorded them.
