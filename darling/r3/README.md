# Round 3 - Darling

This is our round 3 submission (nova_user_fix).

How it works: for every request it picks the live item with the best score, where the score is the item's global appeal (k-adjusted, smoothed) plus a small co-click taste term, plus a big bonus if the user clicked that item before and it's still live. Re-shown past clicks get picked about 67-72% of the time in the R3 logs, so re-click is the main signal. New users just get the top appeal item that's live.

- runs in about 15-30 s on full R3, peak RAM about 1.6 GB
- passed the robustness checks (shuffled rows, unseen users, random timestamps, subsample)
- model.zip sha256: 729605d5ef9d1dd800dd88be77b96ab1353d1c3c9c403bf1c8c3009b47601911
