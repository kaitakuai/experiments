# MiniMax-M2.7: nonce-level tau table (rebuilt from the npz counters in artifacts/)

Points % at tau = 0: mean over hashes [min–max]. Nonce columns: share of nonces with at least one
disagreement whose margin exceeds tau, mean over hashes. Corpora: 250 nonces per block hash.

| validator | prover set | points %, tau = 0 | tau 0.02 | 0.025 | 0.04 | 0.05 | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| B300 | honest, same boot | 0.11 [0.00–0.45] | 0 % | 0 % | 0 % | 0 % | 10 hashes |
| B300 | honest, same card, other boot and plugin build | 6.31 [5.67–6.90] | 7 % | 2 % | 0 % | 0 % | reference artifacts of 7 Sep on this code |
| B300 | honest H200 | 7.32 [6.51–8.10] | 14 % | 5 % | 1 % | 0 % | 10 hashes |
| B300 | honest H100 | 8.03 [7.19–8.67] | 23 % | 12 % | 3 % | 2 % | 10 hashes |
| B300 | honest A100 | 8.11 [7.12–8.92] | 26 % | 14 % | 3 % | 1 % | 10 hashes |
| B300 | fraud QuantTrio, one boot per hash (10 Sep) | 12.10 [11.24–12.93] | 63 % | 41 % | 10 % | 4 % | 26 of 2,500 nonces carry a NaN step |
| B300 | fraud QuantTrio, decayed corpus (7 Sep), every nonce | 11.67 [10.44–13.26] | 31 % | 19 % | 4 % | 2 % | later hashes are NaN chains; not used in the artifact |
| H200 | honest, same boot | 0.28 [0.03–0.61] | 0 % | 0 % | 0 % | 0 % | 10 hashes |
| H200 | honest B300 | 7.36 [6.37–7.89] | 14 % | 6 % | 1 % | 0 % | 10 hashes |
| H200 | honest H100 | 7.60 [6.68–8.30] | 20 % | 10 % | 2 % | 1 % | 10 hashes |
| H200 | honest A100 | 7.82 [6.80–8.65] | 24 % | 13 % | 3 % | 1 % | 10 hashes |
| H200 | fraud QuantTrio, decayed corpus (7 Sep), every nonce | 11.47 [10.00–13.05] | 29 % | 18 % | 4 % | 2 % | later hashes are NaN chains; not used in the artifact |
| H200 | fraud QuantTrio, nonces without NaN steps (h01–h04, 616 nonces) | 12.23 [11.27–13.05] | 66 % | 39 % | 11 % | 4 % | the cell used in the artifact (docs/fig_data_*.json) |
| H100 | honest, same boot | 6.57 [5.64–7.10] | 11 % | 6 % | 2 % | 1 % | 10 hashes |
| H100 | honest B300 | 7.92 [6.99–8.42] | 20 % | 9 % | 1 % | 1 % | 10 hashes |
| H100 | honest H200 | 7.44 [6.65–8.15] | 17 % | 7 % | 1 % | 1 % | 10 hashes |
| H100 | honest A100 | 7.80 [6.83–8.61] | 23 % | 13 % | 3 % | 1 % | 10 hashes |
| H100 | fraud QuantTrio, decayed corpus (7 Sep), every nonce | 11.48 [10.44–12.78] | 29 % | 17 % | 4 % | 1 % | later hashes are NaN chains; not used in the artifact |
| H100 | fraud QuantTrio, nonces without NaN steps (h01–h04, 616 nonces) | 12.11 [11.17–12.77] | 63 % | 37 % | 9 % | 4 % | the cell used in the artifact (docs/fig_data_*.json) |
| A100 | honest, same boot | 2.52 [1.96–2.89] | 0 % | 0 % | 0 % | 0 % | 10 hashes |
| A100 | honest B300 | 8.14 [7.15–8.74] | 26 % | 13 % | 2 % | 1 % | 10 hashes |
| A100 | honest H200 | 7.70 [6.91–8.45] | 22 % | 12 % | 3 % | 1 % | 10 hashes |
| A100 | honest H100 | 8.00 [6.96–8.49] | 27 % | 16 % | 5 % | 3 % | 10 hashes |
| A100 | fraud QuantTrio, decayed corpus (7 Sep), every nonce | 9.63 [8.73–10.80] | 18 % | 10 % | 1 % | 0 % | later hashes are NaN chains; not used in the artifact |
| A100 | fraud QuantTrio, nonces without NaN steps (h01–h04, 616 nonces) | 10.16 [9.51–10.77] | 42 % | 21 % | 3 % | 1 % | the cell used in the artifact (docs/fig_data_*.json) |

Source files: `data/validations/minimax/validator_<card>/npz_<prover>_honest_hNN_v250.npz`,
`npz_honest_hNN_v250.npz` (same boot), `validator_b300/fraud_0910/`, `validator_b300/old_goldens_0907_to_new_code/`.
