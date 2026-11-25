# Recent work (2022–2026) beside ours: a survey, 2026-11-23

Asked for by Kris ("is there more recent papers we can research and cite? … we're at 2026"). Made by a search
agent with generic queries; it reports having opened a page for every entry below (arXiv abstract pages; Europe
PMC records for journals that block automated reading). **Nothing here has been opened by a person yet**: each
paper that enters the draft has a row in docs/paper/REFERENCES.md for Kris to open by hand. The numbers quoted
from papers are the agent's reading and must be checked on the page before they are printed.

## What it changes for us

**The paper's main finding is known from the other side.** Three results a referee will have in mind:
- Thiry, Arbel, Belilovsky & Oyallon (ICLR 2021, arXiv 2101.07528), *The unreasonable effectiveness of patches in
  deep convolutional kernels methods*: one layer of whitened patches sampled from the data and a linear classifier,
  86.7% on CIFAR-10 (78.6% with Gaussian-noise atoms); training the dictionary end to end changes −0.1%;
  whitening "highly affects performance".
- Prabhu, Sinha, Kumaraguru, Torr, Sener & Dokania (NeurIPS 2024, arXiv 2402.08823), *Random representations
  outperform online continually learned representations* (RanDumb): random Fourier features, decorrelation and
  nearest class mean in one pass match or beat continually trained deep networks; without the decorrelation
  CIFAR-10 falls from 55.6 to 30.0, without the random embedding to 41.6.
- Dalm, Offergeld, Ahmad & van Gerven (2024, arXiv 2405.02385), *Efficient deep learning with decorrelated
  backpropagation*, and Offergeld & Ahmad (2024, arXiv 2407.10780), *Correlations are ruining your gradient
  descent*: decorrelating the inputs of every layer speeds up gradient descent itself.
Also Jordan (2024, arXiv 2404.00498): a frozen patch-whitening first layer is the most useful single ingredient
in the fastest CIFAR-10 training. So "the code, not the rule" agrees with a literature we had not read. What is
ours: the same finding reached from a learner built of local rules and taken apart, the like-for-like controls in
a one-pass label stream (gradient readout on the same code; backprop with and without the relay), and the staged
life. The draft says so now (introduction and related work).

**"Hebbian filters do no better than random" is about hard winner-take-all.** Journé, Garcia Rodriguez, Guo &
Moraitis (ICLR 2023, arXiv 2209.11883; SoftHebb) report for one convolutional layer: hard WTA 62.69%, random
weights 63.93%, soft WTA 71.10%. Our rule is the hard one, and our own attempt at the soft one ("my version of
SoftHebb", VISION.md 2026-11-17) was not a faithful replication. A faithful soft competition is owed before the
paper says anything about Hebbian learning in general. Also: Fatan Serj, Parraga & Otazu (2026, arXiv 2604.09734)
80.1% with local plasticity and anti-Hebbian decorrelation against 61.4% with learning off (300 epochs); Wu,
Delrocq, Gerstner & Bellec (ICML 2026, arXiv 2601.21683): with depth, the local rule matters.

**A local, online mechanism for the relay exists in the literature.** Duong, Lipshutz, Heeger, Chklovskii &
Simoncelli (ICML 2023, arXiv 2301.11955), *Adaptive whitening in neural populations with gain-modulating
interneurons*: online whitening with fixed synapses and interneurons that adjust their gains; Duong, Simoncelli,
Chklovskii & Lipshutz (NeurIPS 2023, arXiv 2308.13633): fast gains with slow synaptic plasticity; Lipshutz,
Pehlevan & Chklovskii (2022, arXiv 2209.10634): interneurons make the whitening circuit converge faster. These are
the candidates for queue E 26 (both whitening steps by local rules), beside the lateral network already tried.

**Cautions.** Wadia et al. (ICML 2021): whitening can make information unusable and hurt generalisation.
Karamanlis et al. (Nature 2025): under natural gaze shifts the retina's output is more redundant than
efficient coding predicts (so "the retina whitens" is not the whole story). Warstadt et al. (2023), BabyLM:
curricula mostly did not help language models on child-sized corpora; Feng, Goodman & Frank (EMNLP 2024): global
developmental order did not matter for language; Sheybani et al. (NeurIPS 2023): developmental order helped
vision. Our stages are an untested assumption (the shuffled-order life is owed).

## The list, by topic (✓ = cited in the draft now)

**Local rules without backprop.** ✓ Journé et al. 2023 (2209.11883) · ✓ Jiménez Nimmo & Mondragón 2025, Neural
Networks 190:107628 (2501.17266): hard WTA with lateral inhibition and BCM, 75.2%, ZCA-whitened input, no random
control · Deng et al. 2025, SPHeRe (2510.14810) · Chen, Liu, Laydevant & Grollier 2025, self-contrastive
forward-forward, Nat. Commun. 16:5978 (2409.11593) · Hinton 2022, forward-forward (2212.13345) · Song et al. 2024,
prospective configuration, Nat. Neurosci. 27:348 · ✓ Wu et al. 2026 (2601.21683) · Erdogan, Pehlevan & Erdogan
2025, error broadcast and decorrelation (2504.11558) · Fatan Serj et al. 2026 (2604.09734) · Pinchetti et al.
2024 (2407.01163) · Innocenti et al. 2025 (2505.13124) · Lagani et al. 2023 survey (2307.16236) · Miconi 2021
(2107.01729) · Illing, Gerstner & Brea 2019 (1905.04101): one hidden layer with fixed or unsupervised weights and
a local-rule readout, our shape.

**Decorrelation helps learning.** ✓ Dalm et al. 2024 (2405.02385) · ✓ Offergeld & Ahmad 2024 (2407.10780) ·
✓ Ahmad, Schrader & van Gerven 2022 (2203.13203) · ✓ Dalm, van Gerven & Ahmad 2023 (2310.00965) · Carrigg et al.
2025 (2510.14657) · ✓ Jordan 2024 (2404.00498) · ✓ Wadia et al. 2021, PMLR 139:10617 · ✓ Le Cun, Kanter & Solla
1991, Phys. Rev. Lett. 66:2396.

**Whitening in brains.** ✓ Duong et al. 2023a (2301.11955) · ✓ Duong et al. 2023b (2308.13633) · Lipshutz,
Pehlevan & Chklovskii 2022 (2209.10634) · ✓ Chapochnikov, Pehlevan & Chklovskii 2023, PNAS 120:e2117484120 ·
Karadas et al. 2026, Nat. Neurosci. 29:1109 (olfactory bulb) · Tootoonian et al. 2026, bioRxiv (decorrelation by
gain control, bulb) · ✓ Elsayed, Skyberg, Cang & Meliza 2026, J. Neurosci. 46(1):e1070252025 · ✓ Karamanlis et al.
2025, Nature 637:394 · Duong, Bredenberg, Heeger & Simoncelli 2023 (2305.19869) · Lipshutz & Simoncelli 2024
(2405.17745) · Ruslim et al. 2025, PLoS Comput. Biol. 21:e1013644 · Chini, Pfeffer & Hanganu-Opatz 2022, eLife
11:e78811.

**Random against learned features.** ✓ Thiry et al. 2021 (2101.07528) · ✓ Prabhu et al. 2024 (2402.08823) ·
Gavrikov & Keuper 2023 (2301.11360) · Trockman, Willmott & Kolter 2022 (2210.03651) · Donghi et al. 2025
(2510.06819).

**One pass, continual.** ✓ McDonnell et al. 2023, RanPAC (2307.02251) · ✓ Zhuang et al. 2022, ACIL (2205.14922) ·
✓ Zhuang et al. 2024, F-OAL (2403.15751) · Zou et al. 2025, Fly-CL (2510.16877) · Goswami et al. 2023, FeCAM
(2309.14062) · ✓ Panos et al. 2023, first session adaptation (2303.13199) · Hayes & Kanan 2022 (2203.10681) ·
Ghunaim et al. 2023 (2302.01047) · Soutif-Cormerais et al. 2023 (2308.10328).

**Fast and slow memory.** Bricken et al. 2023 (already cited) · Arani, Sarfraz & Zonooz 2022 (2201.12604) · Sun
et al. 2023, Nat. Neurosci. 26:1438 · Spens & Burgess 2024, Nat. Hum. Behav. 8:526 · Alonso & Krichmar 2024, Nat.
Commun. 15:3722 (a local-rule Hopfield store, online continual) · Lampinen et al. 2025 (2509.16189).

**A child's stream.** ✓ Vong, Wang, Orhan & Lake 2024, Science 383:504 · Orhan & Lake 2024, Nat. Mach. Intell.
6:271 · ✓ Warstadt et al. 2023 (2504.08165) · Hu et al. 2024 (2412.05149) · Charpentier et al. 2025 · ✓ Sheybani
et al. 2023, NeurIPS · Feng, Goodman & Frank 2024 (2408.03617) · Jiang et al. 2026, BabyCL (already cited).

**Dendrites.** Senn et al. 2024, eLife 12:RP89674 · Wright, Hedrick & Komiyama 2025, Science 388:322 (different
plasticity rules in apical and basal dendrites, in vivo) · Galloni et al. 2026, Cell Reports 45:117159 · Greedy
et al. 2022 (2206.11769) · Mikulasch et al. 2022 (2205.05303).

**A model as judge.** ✓ Zheng et al. 2023 (2306.05685) · ✓ Thakur et al. 2025 (2406.12624): only the largest of 13
judges align reasonably with people · ✓ Bavaresco et al. 2025 (2406.18403) · Tan et al. 2025, JudgeBench
(2410.12784) · Ye et al. 2024 (2410.02736). Not to cite: arXiv 2606.19544 (withdrawn by arXiv).

Venues the agent gave from memory, not from the page (check before printing): Ahmad et al. 2022 (TMLR), Lipshutz
et al. 2022 (ICLR 2023), Gavrikov & Keuper (CVPR 2023), Trockman et al. (ICLR 2023), Ye et al. (ICLR 2025).

**Added 2026-11-25 05:09 (read for queue E 27).** Journé et al. 2023 (arXiv 2209.11883, ar5iv rendering to the start of
section 4): the anti-Hebbian term is justified by earlier machine-learning work (Krotov & Hopfield 2019; Grinberg
et al. 2019), not by biology; the per-neuron rate is theirs ("stabilizes to zero as neuron weight vectors
converge to a sphere of radius 1"). ✓ Bienenstock, Cooper & Munro 1982, *J. Neurosci.* 2(1):32–48, doi
10.1523/JNEUROSCI.02-01-00032.1982 (statement read on the encyclopedia page only). ✗ Artola, Bröcher & Singer
1990 (not opened: the record's page did not load).
✓ Blais, Intrator, Shouval & Cooper 1998, *Neural Computation* 10, 1797–1813 (read in the paper's PDF): the
quadratic BCM rule c(c − θ)σ′d with θ = E[c²], stable without normalisation; one cell only. Not opened (server
asks for a browser check): Bienenstock, Cooper & Munro 1982; Law & Cooper 1994 (*PNAS* 91, 7797–7801). To find:
Intrator & Cooper 1992 (the network form with lateral inhibition).

### 2026-11-25 07:40: where BCM's simulation constants are
- Blais, Intrator, Shouval & Cooper (1997), "Receptive Field Formation in Natural Scene Environments: Comparison
  of Single Cell Learning Rules", *NIPS* 10, 423–429: opened (the proceedings' PDF, read locally). Rules and
  input (13 × 13 round patches, 12 natural images, centre-surround or whitened); quadratic BCM "produce[s]
  oriented RFs even when the data is not whitened"; no rate, no time constant.
- B. Blais, *Plasticity* (the authors' simulator, public repository): `plasticity/train.pyx`,
  `plasticity/utils.py`, `plasticity/run_sim.py` read in the source. Defaults: eta 8e-6, tau 1000, weights born
  in [−0.05, 0.05], thresholds in [0.01, 0.02], sigmoid −1 / 50, 500 × 500 presentations, one cell, no lateral
  connections; lateral forms without default strengths. A program's defaults, not a paper's numbers: to be
  cited as the software, with the date read.
- Not opened: Udeigwe, Munro & Ermentrout 2017 (login redirect); a later table of the group (rate 1e-6, first
  threshold 0.7, memory constant 300–1000 in a search result only); Kirkwood, Rioult & Bear 1996; Földiák
  1990; Law & Cooper 1994; Cooper & Scofield 1988.
- Földiák (1990), *Biological Cybernetics* 64, 165–170: opened (scan with a text layer); rules and numbers in
  VISION.md. Falconbridge, Stamps & Badcock 2006 and Zylberberg, Murphy & DeWeese 2011: named from memory, not
  opened.
