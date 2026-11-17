# Vision without backprop: what the literature says about our cortex (survey, 2026-11-17)

A survey by a research agent during the day loop, kept nearly as returned. Tags: **[V]** read in the paper that day,
**[A]** abstract or secondary source only, **[M]** from memory, **[I]** the surveyor's inference. Nothing here is
measured in this project yet.

Our cortex: three convolutional layers of hard winner-take-all competitive learning (64, 128, 256 kinds), response
relu × contrast with only the top kinds kept, read as each layer's map max-pooled to 4×4. A linear readout names
22–35 kinds of things at 0.28–0.55; the first layer alone is about as good as all three; ten times more pictures
gave +0.02.

**The symptom (first layer ≈ all three) is a documented failure of hard winner-take-all stacks, and the same few
fixes recur across papers. Width, dense graded codes, decorrelation and restricted fan-in matter more than the
learning rule or more data.**

## What the literature says

| approach | rule | reported result | what was essential |
|---|---|---|---|
| Coates, Lee & Ng 2011 | k-means on normalised, whitened 6×6 patches; encode with the triangle `f_k = max(0, mean(z) − z_k)` (z: distance to centroid k) | CIFAR-10 77.9% (1,600 features), 79.6% (4,000); hard assignment 68.6% [V] | whitening "crucial"; stride 2 costs ≥3 points; sum-pool over quadrants [V] |
| Coates & Ng 2011 (deep) | the same; each higher unit sees a small group of first-layer maps chosen by correlation | 1 layer 78.3%; 2 layers with full fan-in 77.4%; learned groups 81.2%; 3 layers 82.0% [V] | restricted fan-in over similar features; all layers concatenated [V] |
| Miconi 2021 | hard-WTA `Δw = y(x − w)`, whitened input, 100/196/400 filters | per layer 49.9 / 45.3 / 34.5%; with fixes 58.6 / 64.2 / 64.6% [V] | triangle activation forward, WTA only for plasticity; random sparse connectivity above layer 1; varying k had "minimal impact" [V] |
| Krotov & Hopfield 2019 | winner pulled to the input, the second pushed away (−0.3 to −0.4); steep power readout | fully connected: CIFAR-10 error 49%; convolutional follow-up 23–28% error [V] | the anti-Hebbian push; unit-normalised patches [V] |
| SoftHebb (Journé et al. 2023) | `Δw_k = η·y_k·(x − u_k·w_k)`, `y = softmax(u/τ)`, sign negated for non-winners | CIFAR-10 80.3%, STL-10 76.2%, ImageNet 27.3%; single layer: soft 71.1%, hard WTA 62.7%, random filters 63.9% [V]; CIFAR-100 56.0% [A] | width ×4 per layer (96→384→1536); triangle + power activation 80.3% against relu 70.7%; one epoch is enough [V] |
| Similarity matching (Pehlevan & Chklovskii; Bahroun & Soltoggio) | `y = relu(Wx − My)`; Hebbian forward, anti-Hebbian lateral | CIFAR-10 79.6% (one layer, 4,000 units); two layers only 75.1% [V] | whitening; overcompleteness [V] |
| Latent predictive learning (Halvagar & Zenke 2023) | Hebbian + "predict the next view of the same thing" + decorrelation | layer-local: STL-10 63.2%, CIFAR-10 59.4% [V] | without the predictive term 27% (below pixels) [V] |
| CLAPP (Illing et al. 2021) | Hebbian update gated by a global factor: same object fixated +1, saccade to a new image −1 | STL-10 73.6% fully local; accuracy rises with depth [V] | – |
| Forward-Forward (Hinton 2022) and variants | per layer, activity high for real data and low for negative data | CIFAR-10 54–59%; self-contrastive convolutional variant 80.8% [V] | negative-data design; SoftHebb widths |
| 2025–26 | Gram-matching local loss; multi-frequency plasticity with anti-Hebbian decorrelation; hard WTA + lateral inhibition + BCM | CIFAR-10 81.1% / 80.1% / 75.2% [V/A] | – |

All purely local unsupervised methods end near 80–81% on CIFAR-10 and 56–57% on CIFAR-100, at widths of 1,536–6,144.

## Why our deeper layers add nothing

- Hard-WTA centroids over all channels learn blobs: higher layers become "a less spatially precise version" of
  layer 1 (Miconi) [V]. Our 3×3×64 = 576-dimensional windows are this case.
- A second layer with full fan-in hurt with other rules too (Coates & Ng, Bahroun) [V].
- Hard WTA can be worse than random filters on one layer (SoftHebb) [V]. Running our pipeline with random filters
  is the cheapest diagnostic [I].
- Nearly one-hot top-k codes starve the next layer; dense graded codes pass information on [I].
- Data is not the limit: SoftHebb converges in about 5,000 updates [V], which fits our +0.02 from ten times more.

## Ranked changes

1. **Triangle encoding at inference, WTA only for plasticity**: `relu(u − mean over kinds at that position)^p`,
   p 0.7–1.4, in place of relu × contrast + top-k. Worth +9 points in Coates and in SoftHebb [V].
2. **Whiten the first layer's windows** (and standardise each channel before layers 2 and 3) [V].
3. **Wider, ×4 per layer, stride 1**: something like 256→1024→more, against 64→128→256 [V].
4. **Restricted fan-in for layers 2 and 3**: a group of correlated channels or a fixed random sparse mask [V].
5. **Soft competition with an anti-Hebbian push** in place of hard WTA with a conscience: +8 on one layer [V].
6. **A predictive objective for layers 2 and 3** (two views of the same thing): the only family where depth
   reliably helps [V].
7. **Readout**: average or sum pooling of the dense codes; all layers concatenated [V]; gain for us a guess [I].
8. **No more pictures or epochs** [V/I].

## Organic check (mine, 2026-11-17)

- 1 is lateral inhibition: each neuron's response relative to the average of its neighbours at the same place
  (subtractive normalisation). Organic.
- 2 is what the retina does: centre-surround receptive fields decorrelate natural images (Atick & Redlich 1992).
  Organic, and it belongs before the cortex.
- 3 is more neurons; V1 is far more overcomplete than 64 kinds. It is "size", the third of Kris's questions.
- 4 is local dendritic connectivity: no neuron hears every channel. Organic.
- 5 is graded competition with anti-Hebbian inhibition (Földiák 1990). Organic.
- 6 is learning from the eye's own successive views of one thing; the cortex already has such a rule
  (`Layer.predictive`, `VisualCortex.watch`), not used in lives.

## References

Coates, Lee & Ng 2011 https://proceedings.mlr.press/v15/coates11a/coates11a.pdf ·
Coates & Ng 2011 https://cs.stanford.edu/~acoates/papers/coatesng_nips_2011.pdf ·
Miconi 2021 https://arxiv.org/abs/2107.01729 ·
Krotov & Hopfield 2019 https://arxiv.org/abs/1806.10181 ·
Grinberg, Hopfield & Krotov 2019 https://arxiv.org/abs/1908.08993 ·
Journé et al. 2023 https://arxiv.org/abs/2209.11883 ·
Bahroun & Soltoggio 2017 https://arxiv.org/abs/1702.06456 ·
Halvagar & Zenke 2023 https://www.nature.com/articles/s41593-023-01460-y ·
Illing et al. 2021 https://arxiv.org/abs/2010.08262 ·
Hinton 2022 https://arxiv.org/abs/2212.13345 ·
Chen et al. 2024–25 (SCFF) https://arxiv.org/abs/2409.11593 ·
Löwe et al. 2019 https://arxiv.org/abs/1905.11786 ·
Nøkland & Eidnes 2019 https://arxiv.org/abs/1901.06656 ·
Pinchetti et al. 2024 https://arxiv.org/abs/2407.01163 ·
SPHeRe 2025 https://arxiv.org/abs/2510.14810 ·
Multi-frequency local plasticity 2026 https://arxiv.org/abs/2604.09734 ·
Jimenez Nimmo & Mondragon 2025 https://arxiv.org/abs/2501.17266
