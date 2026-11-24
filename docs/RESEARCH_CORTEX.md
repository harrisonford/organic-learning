# Early visual cortex in numbers, and recent local-rule vision models: a survey, 2026-11-23

Asked for by Kris: the cortex's sizes "out of a 2011 recipe" should rest on what is known of the brain and on
recent work. Made by a search agent with generic queries; it reports a page opened for every source (Europe PMC
records where a journal blocks reading). Not yet opened by a person. Standing rule 8 applies: biology decides,
not the score. docs/RESEARCH_VISION.md (2026-11-17) is the earlier survey; this one adds the numbers of biology.

## A. What biology says

**Expansion.** Retina → LGN is about 1:1 (macaque; Schein & de Monasterio 1987). LGN → V1 is a large expansion:
356 V1 neurons per LGN neuron in humans, 87 in the tarsier, V1 neurons growing as LGN^1.5 (Stevens 2001, Nature
411:193); about 25:1 in the cat (Olshausen & Field 2004); V1's dimensionality about 30 times the LGN's (DiCarlo,
Zoccolan & Rust 2012). Some 343,000 neurons under 1 mm² of macaque V1 (Garcia-Marin, Kelly & Hawken 2024). A
hypercolumn (0.5–1 mm; all orientations, both eyes, 2 blobs) holds every value of every receptive-field variable
(Horton & Adams 2005; Swindale et al. 2000): of the order of 10^5 neurons.

**Receptive fields.** Diameter about 0.21 × eccentricity in V1, 0.46 in V2, 0.84 in V4 (Freeman & Simoncelli
2011); IT 10–25° (DiCarlo et al. 2012). A simple cell has 2–3 ON/OFF subfields; macaque fields have fewer
subfields than sparse coding predicts and many are blob-like (Ringach 2002). At one place the preferred spatial
frequency spans a wide range (De Valois et al. 1982): several scales at every location.

**Pooling.** Complex cells sum the squared outputs of filters of one orientation and different phase (the energy
model; Rust et al. 2005, who also find suppressive filters in 41 of 59 cells). A complex field is about twice a
simple one, pooled within orientation and neighbouring scales (Serre et al. 2007; Chen et al. 2020). Tolerance
then grows stage by stage (IT: shifts of 1.5–3°, sizes 2–3×), and is reshaped by what follows what in time within
an hour (Li & DiCarlo 2008).

**Normalisation.** The established form is divisive: R = Dⁿ / (σⁿ + Nⁿ), n between 1 and 3.5, about 2, the pool
wider than the field in space and in tuning (Carandini & Heeger 2012); the surround acts as a divisive gain
(Cavanaugh, Bair & Movshon 2002); fitted on natural images the pool is orientation-specific and mostly inside the
field, exponent about 1.6 (Burg et al. 2021). Subtraction exists too: PV interneurons divide, SOM interneurons
subtract (Wilson et al. 2012).

**Sparseness.** Lifetime sparseness 0.41 with the field alone stimulated, 0.52–0.62 with the surround too (Vinje
& Gallant 2000); medians 0.88 / 0.81 / 0.80 in V1 / V2 / V4 (Willmore, Mazer & Gallant 2011); about 0.5% of
layer-2 neurons respond above half their maximum to a natural image (Tang et al. 2018).

**No weight sharing.** Nothing copies weights; the set of features is repeated statistically in every
hypercolumn, and not uniformly (scale grows with eccentricity). Networks without sharing lose 7 points on CIFAR-10;
lateral connections with a sleep-phase Hebbian step recover most of it (Pogodin et al. 2021); exposure to
translation makes unshared filters converge (Ott et al. 2020).

**Colour.** About 60% of macaque V1 cells prefer luminance, 29% are colour-luminance (oriented, double-opponent),
11% colour-preferring (unoriented, low-pass) (Shapley & Hawken 2011); colour and orientation are coded jointly
(Garg et al. 2019).

**Before experience.** Retinotopy and orientation selectivity form from molecular cues and spontaneous activity;
experience maintains and matches them (Espinosa & Stryker 2012); direction selectivity needs moving stimuli (Li et
al. 2008). The same coding rule run on wave-like spontaneous patterns gives oriented, localised filters (Albert,
Schnabel & Field 2008).

**The retina.** The LGN whitens in time (Dan, Atick & Reid 1996); luminance and contrast gain controls are
separate and divisive (Mante et al. 2005); most of the retina's decorrelation comes from a threshold
nonlinearity, not from the linear field (Pitkow & Meister 2012). Exact linear whitening is stronger than what the
retina does.

## B. Local-rule vision models (CIFAR-10, linear readout unless said)

| Model | Shape | Rule | Accuracy |
|---|---|---|---|
| Coates, Lee & Ng 2011 | 1 layer, 6×6 whitened patches, mean-subtract rectification, 2×2 quadrant sum | k-means | 77.9 (1,600), 79.6 (4,000) |
| Bahroun & Soltoggio 2017 | 1 layer, whitened, quadrant pooling | Hebbian/anti-Hebbian, lateral inhibition | 79.6 (4,000); 80.4 with 4×4, 6×6, 8×8 side by side |
| Grinberg, Hopfield & Krotov 2019 | 1 layer, 4×4 patches, 400 channels, ReLU^40, max-pool | rank-based competitive Hebbian | 72.2; 76.6 with mixed patch sizes |
| Lagani et al. 2020 | 4 conv layers | hard WTA; Hebbian PCA | WTA 63.9 → 53.0 with depth; PCA about 64 |
| Miconi 2021 | 3 layers (100/196/400), whitened | instar with WTA | 49 → 35 with depth; 59 → 65 with mean-subtract rectification and 1% connectivity |
| Journé et al. 2023 (SoftHebb) | 3 layers (5×5, 3×3, 3×3; 96/384/1536), mean-subtract then a power, max-pool | soft WTA, Δw = η·y·(x − u·w) | 80.3; 1 layer 71.1 |
| Illing et al. 2021 (CLAPP) | VGG-6, STL-10 | layer-wise contrastive predictive | STL-10 73.6 |
| Halvagal & Zenke 2023 (LPL) | VGG-11 | Hebbian plus predictive, layer-local | 59.4 |
| Hinton 2022 (forward-forward) | 2–3 locally connected layers, 11×11 fields, no sharing | goodness, supervised | 59 |
| Chen et al. 2025 (self-contrastive FF) | SoftHebb's shape | unsupervised goodness | 80.8 |
| Jiménez Nimmo & Mondragón 2025 | 3 layers, whitened | hard WTA, centre-surround lateral inhibition, BCM | 75.2 (19.9 without competition) |
| Ravichandran et al. 2024/25 (BCPNN) | no convolution; hypercolumns of 100 minicolumns | Hebbian plus structural plasticity | 61.2 |

Across them: width is the reliable lever (more features, denser stride); depth hurts under plain competitive rules
and helps only with more width per layer (SoftHebb: ×4), sparse connectivity (Miconi), or a predictive objective
(CLAPP, LPL); small fields win on 32×32 pictures (4–7) and mixed sizes help.

## C. Our cortex against this

| What we have | Against biology | What biology supports instead |
|---|---|---|
| 5×5 patches, one size | a small field with 2–3 subfields is fine; one scale is not | several patch sizes at each place |
| 256 filters | not too many: V1 is 25–356 times its input | no fixed number; more, when the machine allows |
| exact whitening of patches (batch) | stronger than the retina, and not how it does it | centre-surround ON/OFF and colour-opponent channels, local gain control, a threshold |
| response above the mean of all filters, rectified | subtractive inhibition exists, but the canonical form is divisive, and ours is far too dense (about half the units on; V1: 0.5% strongly) | divisive normalisation, exponent about 2, a pool of similar tuning reaching into the surround |
| average over a 4×4 grid, in one step | a jump from V1-sized to V4/IT-sized fields, and not the energy form | sum of squares over phase and about twice the field, within a feature; then stages that each about double the field; tolerance learned from what follows what |
| the same filters at every place | an approximation of the repeated hypercolumn; the mechanism is not biological | units without sharing that learn the same statistics, evened out laterally or in sleep |
| one layer | V1 alone is two stages after a retina/LGN stage | more stages; plain competition loses information with depth unless width grows, wiring is sparse or the objective predicts |
| filters start random, learn on photographs | orientation is there before the eyes open | the same rule first run on wave-like spontaneous patterns, then on pictures |

Unverified in the survey: the macaque LGN:V1 ratio; a simple cell's width in cones at the fovea.
