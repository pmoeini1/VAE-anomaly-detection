# VAE Anomaly Detection on Hourly Electricity Consumption
 
An anomaly detector for a time series, built with a variational autoencoder (VAE) in PyTorch. It uses a dense VAE over sliding windows, scored by reconstruction probability.
 
## How it works
 
1. **Data:** one client's hourly electricity consumption from the Hugging Face dataset, streamed with `datasets`.
2. **Split:** the first 80% of the series is used for training and validation, and the last 20% is held out for testing.
3. **Cleaning the training data:** a median weekly profile (168 hourly slots) is subtracted from the series, and points whose residual falls outside 1.5×IQR are flagged. Any 48-hour window containing a flagged point is dropped from training.
4. **Model:** an MLP VAE. The encoder maps a 48-hour window to an 8-dimensional latent distribution, and the decoder predicts a mean and variance for every hour.
5. **Scoring:** the anomaly score is the negative log-likelihood of a window under the decoder, averaged over 20 latent samples (reconstruction probability). Higher means more anomalous.
6. **Evaluation:** test windows are labeled with the same weekly-residual IQR rule, using statistics computed on the training data only. The detection threshold is the 99th percentile of clean validation scores, never tuned on the test set.

## Results

| Metric | Value |
|---|---|
| Mean anomaly score, clean validation windows | -21.3 |
| Mean anomaly score, windows removed from training | 1529.6 |
| Threshold (99th percentile of clean validation scores) | 23.44 |
| Precision | 0.865 |
| Recall | 0.817 |
| F1 | 0.840 |

Anomaly score is the negative log-likelihood of a window, so lower (including negative) values mean the window looks normal.

## Limitations

- The test labels come from the same weekly-residual IQR rule used to clean the training data, so these metrics measure how closely the VAE agrees with that rule, not how well it detects real faults.
- Results are from a single client's series (the first in the dataset) and a single train/test split.
- Each flagged hour marks every 48-hour window containing it as anomalous, so the window-level labels are strongly correlated.