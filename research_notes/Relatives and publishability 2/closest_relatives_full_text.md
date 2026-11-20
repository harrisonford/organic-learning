# Closest relatives: full-text read and missed-work search

Date of reading: 2026-11-19. Follows the first survey in `Backprop free developmental learning survey/` and does not repeat it.

**Markers.** [V] verified this session in the paper's full text or on a fetched page. [S] seen only in a search snippet. [R] recalled from memory. [D] my arithmetic from [V] numbers.

**How the reading was done.** arXiv HTML full texts were downloaded and converted to plain text. Four papers (2306.15364, 2507.11788, 2503.02057, 2607.26523) were read end to end directly. Four (Dragon Hatchling, Monty, Sparse Distributed Memory, SoftHebb) were read end to end by sub-readers working from the same text files, with their reports checked against the Semantic Scholar and OpenAlex counts fetched here. Figures were not visible in any of them, so numbers shown only inside plots are missing.

---

## 0. What the reading changes

1. **None of the eight "closest relatives" does what this system does: name held-out real photographs with a learner that has no backpropagation anywhere and is trained once through.** The two assembly-calculus papers and the fast-mapping spiking model use pre-formed symbolic input patterns, not images. SoftHebb, the Sparse Distributed Memory learner and the Dragon Hatchling all use backpropagation somewhere (a supervised readout, the whole network, and the whole network respectively). The Sparse Distributed Memory learner and the fruit-fly learner also both sit on features from a network pretrained with labels by backprop. The first survey already noted backprop in the Sparse Distributed Memory learner and left the Dragon Hatchling's training "not confirmed from its full text"; that is now confirmed (truncated backprop through time, Appendix B.1 [V]).
2. **The first survey missed one whole research line that is the nearest neighbour in spirit**: the "brain-constrained" Hebbian spiking models of word learning from the Berlin group (Tomasello et al. 2018; Constant, Pulvermüller and Tomasello 2023). The 2025 assembly-calculus paper itself names that line as its "very interesting precursor" [V]. The 2023 paper is a Hebbian account of fast mapping that depends on a prior babbling-like stage, which is exactly this system's developmental argument.
3. **It also missed the nearest backprop comparator for the single-pass claim**: BabyCL (Jiang, Yang, Norman, Lake and Ren, arXiv 2606.05115, June 2026), which trains the child's-view contrastive model in one chronological pass. Its naive one-pass baseline falls to chance, and its fix is a replay buffer holding 60% of the stream. This is the paper a reviewer will reach for when reading "0.52 after one pass against 0.39 for backprop".
4. **Reception of the closest non-backprop language papers is thin.** Both assembly-calculus language papers are still preprints with 0 to 4 citations; the single-author Hebbian language paper has 0 to 3; the 2026 delta-rule continual learner has 0. Being first in this niche has not, so far, earned attention by itself.
5. **The eight papers set a low bar on statistics and a clear bar on honesty.** Seeds range from one to five; several report no variance at all. The ones that were received well (SoftHebb, Sparse Distributed Memory) had standard benchmarks, published baselines in a table, and ablations. The ones that drew public criticism (Dragon Hatchling) were criticised for one baseline, no ablations and biological over-claiming.

---

## 1. The eight papers

### 1.1 Summary table

| Paper | Learning rule in fact | Data | Runs | Baselines | Venue | Citations |
|---|---|---|---|---|---|---|
| Language organ, arXiv 2306.15364 | Hebbian multiplicative update plus k-winners; no backprop [V] | Toy 2-word language; pre-formed assemblies as "vision" [V] | 5 per condition [V] | None [V] | Preprint, v1 only [V] | 1 (Semantic Scholar), 4 (OpenAlex) [V] |
| Simulated acquisition, arXiv 2507.11788 | Same [V] | Toy 2- and 3-word sentences; pre-formed assemblies [V] | 10 and 20 per condition [V] | None [V] | Preprint; also on bioRxiv [V] | 0 [V] |
| Hebbian local structure of language, arXiv 2503.02057 | Hebbian n-gram counts with thresholds [V] | *Alice in Wonderland* text; random languages [V] | None reported [V] | None [V] | Preprint, v1 only [V] | 0 (Semantic Scholar), 3 (OpenAlex) [V] |
| CMP continual learner, arXiv 2607.26523 | Delta rule on a linear readout over fixed random sparse codes [V] | 15 text domains, byte level; Split-MNIST [V] | 3 seeds for own model; baselines appear single-run [V] | Transformer, naive and online EWC [V] | Preprint, v1 only [V] | 0 [V] |
| Dragon Hatchling, arXiv 2509.26507 | **Backprop through time** for all parameters; Hebbian only as in-context state [V] | Europarl translation, byte level [V] | None reported for main result [V] | One GPT-2-style Transformer [V] | Preprint, v1 only [V] | 7 (Semantic Scholar), 10 (OpenAlex) [V] |
| Monty, arXiv 2507.04494 | Supervised storage of points in object reference frames; no gradients [V] | 77 simulated YCB objects; no real images [V] | None reported [V] | Vision Transformer, pretrained and from scratch [V] | *Neural Computation* 38(6), 2026 [V] | 5 (Semantic Scholar), 6 (OpenAlex) [V] |
| Sparse Distributed Memory learner, ICLR 2023 | **Backprop** (plain SGD) on a one-hidden-layer k-winners network [V] | Split CIFAR-10/100 on frozen ImageNet-pretrained embeddings; Split MNIST from pixels [V] | 5 task splits or 3 seeds [V] | EWC, MAS, SI, fruit-fly model and five others [V] | ICLR 2023 poster; scores 5, 6, 8, 8 [V] | 26 (Semantic Scholar), 28 (OpenAlex) [V] |
| SoftHebb, ICLR 2023 | Local soft winner-take-all Hebbian conv layers; **supervised gradient-trained linear head** [V] | MNIST, CIFAR-10, STL-10, ImageNet [V] | 4 (1 for ImageNet) [V] | 11 published local-rule methods plus backprop [V] | ICLR 2023, notable top 25% [V] | 99 (Semantic Scholar), 81 (OpenAlex) [V] |

Citation sources: [Semantic Scholar batch API](https://api.semanticscholar.org/graph/v1/paper/batch) and [OpenAlex](https://api.openalex.org/works), both queried 2026-11-19.

### 1.2 "The Architecture of a Biologically Plausible Language Organ" (Mitropolsky and Papadimitriou) — [arXiv 2306.15364](https://arxiv.org/abs/2306.15364)

- **Claims.** A simulated language organ "succeeds in an important early step in language acquisition: the learning of nouns, verbs, and their meanings, from the grounded input of only a modest number of sentences. Learning in this system is achieved through Hebbian plasticity, and without backpropagation" [V]. Framed as "a proof of concept, an existence theorem", explicitly not a theory to test on humans [V].
- **Model.** Brain areas of n neurons with random connections; at each step the k most-driven neurons fire; a synapse is multiplied by (1+β) when its two ends fire in succession [V]. Areas: Phon, LexN, LexV, Visual, Motor and C context areas [V].
- **Input.** A toy language of ℓ nouns and ℓ intransitive verbs; every sentence is two words; word order may be noun-verb or verb-noun [V]. "Seeing" is the activation of a **pre-initialised assembly** in Visual; hearing is a pre-initialised assembly in Phon [V]. There are no images, no sounds, no text corpus.
- **Parameters.** Text example n=10^6, k=10^3, β=0.1, ℓ=5, τ=2; figure runs n=10^5, p=0.05, β=0.06, k=50 in the lexical areas, C=20 [V].
- **Success measure.** Not accuracy on held-out items. A run succeeds when every word satisfies property P (firing the meaning recovers at least 75% of the word's Phon assembly) and Q1 to Q3 (stable assembly in the correct lexical area, unstable in the other) [V]. The result reported is the number of training sentences until first success.
- **Results.** Sentences needed grow roughly linearly with lexicon size, "10-20 overhead sentences per word" [V]. Interleaving single-word tutoring "reduced the training time by over 40%" [V]. Higher β speeds convergence [V].
- **Statistics.** "Each experiment is repeated 5 times; means and standard deviations are reported" [V]. No baselines, no ablations beyond β and context areas, no significance tests [V].
- **Length.** arXiv comment "9 pages, 4 figures" [V]; about 5,900 words of body [D]; 6 sections; 2 figures in the HTML body.
- **Admitted limits.** The number of training sentences "is rather large, and scales linearly with the size of the lexicon"; phonology is abstracted away; syntax, generation, function words, abstract words and multilinguality are all listed as future work; the synchrony assumption "is definitely unrealistic" [V].
- **Status and reception.** One arXiv version, 27 Jun 2023, cs.CL and q-bio.NC; no journal reference [V]. No published critique or review found [S].

### 1.3 "Simulated Language Acquisition in a Biologically Realistic Model of the Brain" (Mitropolsky and Papadimitriou) — [arXiv 2507.11788](https://arxiv.org/abs/2507.11788)

- **Claims.** "the first biologically plausible system that can learn the rudiments of natural language — lexicon as well as basic syntax and semantics — through the presentation of a modest amount of grounded sentences in any language" [V]. Learns (a) meanings of concrete nouns and verbs, (b) noun versus verb, (c) word order in several moods, (d) generation of novel sentences [V]. "it does not use backpropagation" [V].
- **Positioning sentence worth quoting.** "In a very interesting precursor to this work, it was shown in [37] that concrete nouns, verbs and their meanings can be learned through grounded presentation in a different biologically realistic model of the brain. However, that work was about the easier problem of learning isolated words" [V]; reference 37 is Tomasello, Garagnani, Wennekers and Pulvermüller 2018 [V].
- **Setup, words.** Same toy two-word language and pre-initialised Phon, Visual and Motor assemblies as the 2023 paper; context areas m from zero to ten [V].
- **Setup, syntax.** Adds four Role areas, three syntactic areas and a Mood area, 18 fibres [V]. The thematic roles of each scene (agent, action, patient) are **assumed already computed** and presented as assemblies [V]. Mood assemblies are given [V].
- **Results.**
  - About 10 sentences per word to satisfy three formation properties, linear in lexicon size [V].
  - The wrong-area assembly is unstable: with "dog" paired with 7 verbs, 30% stability in the verb area against 100% in the noun area [V].
  - Word learning tolerates transitive sentences up to "roughly a half" of input [V].
  - Word order: success means correct generation of one withheld transitive and one withheld intransitive sentence [V]. Sentences needed grow roughly linearly with number of moods (10 runs each) [V]. Object-first orders need "significantly more sentences" (20 runs per order, 70% transitive input) [V]. With under half transitive input, three orders "cannot be learned by this system even with unbounded input" [V].
  - Sequences of assemblies are hard to hold beyond 20 to 40 items [V].
- **Statistics.** Run counts only; no baselines, no error analysis in text, no tests [V].
- **Length.** arXiv comment "13 pages, 6 figures" [V]; about 8,000 words of body [D]; three sections (Introduction; Procedures and Results 2.1 to 2.6; Discussion) plus supplementary information; 38 references [V].
- **Admitted limits.** "The learning of speech sounds and phonology has been bypassed, while adjectives, adverbs, and functional categories are not considered"; abstract words need a new representation scheme; hierarchy is sketched, not built; the Mood assemblies need "more work … to fully justify" [V].
- **Status and reception.** One arXiv version, 15 Jul 2025, cs.NE and cs.CL [V]; a bioRxiv copy exists (DOI 10.1101/2025.07.15.664996) [V]; described as under review [S]; an earlier COSYNE 2025 poster [S]. No citations yet on either index [V].

### 1.4 "Hebbian learning the local structure of language" (single author, physics) — [arXiv 2503.02057](https://arxiv.org/abs/2503.02057)

- **Claims.** A hierarchy of Hebbian neurons "learns to tokenize words from text", plus added neurons that bind replayed patterns into an embedding; "permits continuous parallel learning without forgetting"; "can learn a natural language morphology without data" [V].
- **What it is.** A theory paper. The model is an n-gram hierarchy: Hebbian counts of adjacent tokens, a threshold ε per level to decide which n-grams become units, and a "smoothness" constraint that every n-gram be made of learned (n−1)-grams [V]. "Replay" fires an auxiliary neuron while the hierarchy generates a word, tying the word's n-grams to that neuron [V].
- **Data and evidence.** *Alice in Wonderland* at four sizes up to 107,777 characters and 2,567 unique words; a random language of 429 words grown from uniform random weights with ε2=0.7, ε3=0.85, ε4=0.45 [V]. Evidence is the match between n-gram count distributions and a log-normal curve [V].
- **Statistics.** None. No task metric, no held-out data, no baseline, no seeds [V].
- **Vision and words.** Only a thought experiment: "say girl and imagine a girl" [V].
- **Length.** arXiv comment "10 figures, 14 pages" [V]; about 8,300 words including appendices [D]; 9 sections and 5 short appendices.
- **Admitted limits.** The hierarchical model "suffers from a combination of forgetting and poor scaling"; how the projection connections are learned "is a point we leave to future research"; the pinning field "is not meant to be taken literally"; "A theory for how measurement arises biologically is outside the scope of this work" [V].
- **Status and reception.** One arXiv version, 3 Mar 2025; no journal reference [V].

### 1.5 "The Art of Not Forgetting: A Local Learning Architecture for Continual Learning" (CMP) — [arXiv 2607.26523](https://arxiv.org/abs/2607.26523)

- **Claims.** Deliberately narrow: "on a controlled domain-incremental protocol, with matched architecture size and matched data budget, CMP's backward transfer is substantially better than a Transformer trained with online EWC" [V]. "It does not claim CMP is more accurate than a Transformer; it is not, by a substantial margin" [V]. "None of these individual components is presented as novel" [V].
- **Model.** Fixed random byte embeddings, multiplied pairwise, k-winners sparsified; a two-tier memory (fast buffer, slow register) with thresholded writes; a locally trained predictor; a **linear readout trained by the delta rule** (error times input); per-tensor slowing of updates by accumulated weight movement [V]. Representation 1,024, k=64, buffer 64, register 256, learning rate 0.12, batch 64, 6,000 steps per domain [V]. So it is mini-batch, not sample-by-sample.
- **Data.** 15 text domains assembled by the authors (Wikipedia, CPython, Shakespeare and so on), byte-level next-byte prediction, no replay, no task labels [V]. "does not constitute a standardized continual-learning benchmark" [V].
- **Results.**
  - Three seeds, 150,000-byte budget: backward transfer +0.2406 ± 0.0092 bits per byte [V].
  - Matched comparison at 3.8M bytes, about 6.5M parameters: naive Transformer +2.7897, online EWC +2.2457, CMP +0.1482, "94.7% reduction" [V]. No seed count is given for these three rows [V].
  - Domain order, five-domain subset: +0.2353, +0.4442, +0.3343 [V].
  - Split-MNIST: dense −0.0211 ± 0.0004, sparse −0.0207 ± 0.0003, reported as a null result [V].
  - Single-domain quality 3.1 to 3.3 bits per byte; the Transformer's figure is never given [V]. Combining with a deeper variant made it worse, 3.48 against 3.27 [V].
  - 6.4M to 29.8M parameters moved backward transfer only from +0.1482 to +0.1397 [V].
- **Things a reviewer would catch.** The five-domain canonical figure in the order table (+0.2353) is identical to the 15-domain seed-42 figure; the baseline section appears twice with the same heading; limitations are referred to as "Section 6" but are Section 5 [V].
- **Length.** arXiv comment "28 pages, 11 figures" [V]; about 7,700 words with appendices [D]; 6 sections, 7 appendices, 17 references [V].
- **Admitted limits.** Accuracy gap; custom benchmark; one continual-learning baseline; order sensitivity tested on 3 of 120 orders; no gain on Split-MNIST; code "not publicly available" [V].
- **Status and reception.** One arXiv version, 29 Jul 2026, cs.LG; two authors at a small company; no citations [V].

### 1.6 "The Dragon Hatchling" (Kosowski et al.) — [arXiv 2509.26507](https://arxiv.org/abs/2509.26507)

- **The fact that matters for positioning.** It is **not** a backprop-free learner. "We train all models using Truncated Backpropagation Through Time" (Appendix B.1) [V]. The Hebbian element is the in-context attention state, "sometimes called 'fast weights'", which in the trained model is "never materialize[d]" and is computed as linear attention [V]. The graph and spiking form of the model is never trained [V].
- **Claims.** "BDH rivals GPT2-architecture Transformer performance on language and translation tasks, at the same number of parameters (10M to 1B)"; "The BDH model is biologically plausible" [V].
- **Setup.** Europarl, raw bytes, translation plus language modelling; 1.2B tokens, about 3 epochs; models 25M to 800M parameters (the abstract says 1B) [V]. One baseline, a GPT-2-style Transformer [V]. The match claim is for a variant with extra gating, shown as a plot with no table [V].
- **Statistics.** No seeds or error bars for the main result; one significance test in the whole paper (a synapse-level interpretability check, 50 against 50 sentences, p < 1e-14) [V]. Structure analysis used 5 seeds on a 24M model [V].
- **Removing backprop through time** ("preliminary"): loss rose from about 0.65 to about 0.75 to 1.05 and the model "lost the ability to match concepts between different languages" [V].
- **Length.** 8 sections plus 5 appendices; about 27,000 words of main text and 6,500 of appendix; roughly 70 to 75% theory by the reader's estimate; 17 figures, 6 tables [V].
- **Admitted limits.** "we did not optimize the BDH-GPU model for suitability of training without backpropagation"; grounding in brain science "was not the objective of this paper" [V].
- **Status.** One arXiv version, 30 Sep 2025; no venue found [V].
- **Reception.**
  - Attention far beyond its citations: official repository 3,582 stars [V]; [Hacker News thread](https://hn.svelte.dev/item/45668408) 134 points [V].
  - Criticisms on that thread: "exactly one comparison with transformer architecture (Figure 7)"; comparison against "the antiquated GPT2"; the abstract trips a "'crank' radar" [V].
  - A [forum post](https://discourse.numenta.org/t/bdh-baby-dragon-hatchling/12185) says of the released code "there is no hebbian learning or plasticity" and it "uses gradient descent (Adam)" [V].
  - An [independent short replication](https://huggingface.co/SlayerLab/bdh-scaling) found faster early learning than four baselines but ran only 4,000 steps with no seeds [V].
  - Authors' follow-up arXiv 2608.09888 (Aug 2026) reports 29.5% on public ARC-AGI-1 with 150M parameters [V].
  - No peer-reviewed paper building on or rebutting it was found.

### 1.7 "Sparse Distributed Memory is a Continual Learner" (Bricken et al., ICLR 2023) — [arXiv 2303.11934](https://arxiv.org/abs/2303.11934)

- **The fact that matters for positioning.** It is **trained with backprop**. The work makes Sparse Distributed Memory trainable "in the deep learning framework (this includes having no fixed neuron weights and using backpropagation)"; backprop is used "as a more efficient (although possibly non-biological ...) implementation of Hebbian learning rules" [V]. The paper itself says of its rival: "The FlyModel is unique in being trained for only one epoch and not using backpropagation" [V].
- **Claims.** "every component of our MLP variant translated from biology is necessary for continual learning"; "free from any memory replay or task information"; combined with EWC, state of the art on class-incremental CIFAR-10 without replay [V]. It concedes that the fruit-fly model "appears to be state-of-the-art for CIFAR-100, MNIST, and FashionMNIST, with our SDMLP as a close second" [V].
- **Model.** One hidden layer; k-winners activation; positive weights; L2-normalised inputs and weights; no bias terms; plain SGD without momentum; a gradual switch from all-active to k-active during pretraining [V].
- **Setup.**
  - Split CIFAR-10, class-incremental, 5 tasks of 2 classes [V].
  - Inputs are 256-dimensional embeddings from a **frozen ConvMixer pretrained with labels on ImageNet32**; the layer itself is also pretrained on those embeddings [V].
  - 1,000 or 10,000 hidden neurons [V].
  - **2,000 epochs per task** on CIFAR-10, 500 on CIFAR-100 and MNIST [V].
  - 5 random task splits for CIFAR-10, 3 seeds elsewhere; standard error only in one figure, no variance in tables, no tests [V].
- **Results** (final accuracy, 10,000 neurons, CIFAR-10 embeddings) [V]: SDMLP 0.71; fruit-fly model 0.82; MAS 0.67; EWC 0.65; plain ReLU 0.21; SDMLP plus EWC 0.86; all-data oracle 0.93.
  - From raw pixels: SDMLP 0.42, oracle 0.53 [V].
  - Split MNIST from pixels: SDMLP 0.69 (1,000 neurons), fruit-fly model 0.77 and 0.91, SDMLP plus EWC 0.83 and 0.86 [V].
  - CIFAR-100 at 10,000 neurons: SDMLP 0.43, fruit-fly model 0.58 [V].
- **Ablations** (1,000 neurons; chance about 0.20) [V]: full model 0.63; with momentum 0.54; Adam 0.23; no gradual switch 0.35; negative weights allowed 0.57; no L2 normalisation 0.20; any bias term 0.20; plain k-winners without the other changes 0.29.
- **Baselines.** ReLU, L2, dropout, plain k-winners, EWC, MAS, SI, fruit-fly model, NISPA, Active Dendrites [V].
- **Length.** arXiv comment "9 Pages"; the PDF is 57 pages with appendices A to H, 29 figures and 10 tables; roughly 5,800 words of main text against 17,000 of appendix [V].
- **Admitted limits.** "Our biggest limitation remains ensuring that SDM successfully avoids the dead neuron problem"; training inside or on top of a jointly trained network "resulted in many dead neurons and failure to continually learn"; "We use preprocessed image embeddings because single hidden layer MLPs struggle to learn directly from image pixels"; sparsity "is also suboptimal for maximizing classification accuracy in non continual learning settings" [V].
- **Peer review.** ICLR 2023, "Accept: poster" [V] ([ICLR page](https://iclr.cc/virtual/2023/poster/10761)). Scores 5, 6, 8, 8 (mean 6.75), confidences 3, 4, 3, 4, read from public mirrors of the review data because OpenReview blocked direct access [V].
  - Reviewer at 5: "the technical novelty of the paper reduces to the use of a top-k activation function in MLP hidden layers"; "the setting is quite arbitrary... given that imagenet embeddings already separate CIFAR10 classes" [V].
  - Reviewer at 6: "Too many modifications to SDM make the final method no more SDM"; "the amount of tuning hyperparameters... is enormous"; only better than the fruit-fly model on CIFAR-10 [V].
  - Reviewers at 8: "Evaluation done only on one benchmark"; "Differences from (Shen et al. 2021) not clear"; wanted other modalities and more layers [V].
  - Area chair, why not higher: "It would be good if method worked for full network (future work)"; why not lower: "Good results and interesting paper" [V].
- **Reception.** Cited by a 2025 *Neuron* review on key-value memory and by spiking and associative continual learners (titles only) [V]. A short follow-up, "Hard ASH" (arXiv 2404.17651), reports 78.3% ± 1.4 on the same Split-MNIST network after **one** epoch with Adagrad, against the paper's 69% after 500 epochs [V]. No formal rebuttal found.

**The fruit-fly model** (Shen, Dasgupta and Navlakha, [arXiv 2107.07617](https://arxiv.org/abs/2107.07617); *Neural Computation* 2023, "Reducing Catastrophic Forgetting With Associative Learning: A Lesson From Fruit Flies" [V]) is the true local-rule relative in this pair.
- **Rule.** A fixed sparse random expansion with k-winners ("there are no trainable parameters in the first layer"), then an associative update of only the target class's output weights; "does not require backpropagation" [V].
- **Exposure.** Class-incremental, **a single pass** [V].
- **Data.** MNIST plus FashionMNIST as 10 two-class tasks on 84-dimensional features from a LeNet trained on another dataset; CIFAR-100 as 25 four-class tasks on 512-dimensional features from an ImageNet-pretrained ResNet18 [V]. So the features come from backprop.
- **Results.** After 5 tasks 0.86 ± 0.0006 against generative replay 0.77, GEM 0.69, EWC 0.58; at least 0.15 above all continual-learning baselines on CIFAR-100; 0.11 to 0.13 below offline training [V]. Five initialisations, mean and standard deviation [V].
- **Ablation.** The same sparse code with a logistic-regression readout drops from 0.75 to 0.54 and from 0.41 to 0.21 [V], so the associative freeze-the-others rule carries the effect.
- **Citations.** 17 to 18 [V].

### 1.8 "Hebbian Deep Learning Without Feedback" — SoftHebb (Journé et al., ICLR 2023) — [arXiv 2209.11883](https://arxiv.org/abs/2209.11883)

- **Claims.** "an algorithm that trains deep neural networks, without any feedback, target, or error signals" [V]. "With up to five hidden layers and an added linear classifier, accuracies on MNIST, CIFAR-10, STL-10, and ImageNet, respectively reach 99.4%, 80.3%, 76.2%, and 27.3%" [V].
- **What is local and what is not.** The convolutional layers learn by a soft winner-take-all Hebbian rule, local in space and time [V]. The classifier is not: "a linear classifier head was trained with cross-entropy loss, using dropout regularization, with mini-batches of 64 examples, and trained for 50, 50, 100, and 200 supervised epochs" [V]. Batch normalisation precedes every layer; weights are shared across positions, which the authors call "not biologically plausible" in the literal sense; hyperparameters were grid-searched per layer on labelled validation accuracy [V].
- **Exposure.** "All results we report on SoftHebb are in fact after just one epoch of unsupervised learning"; ImageNet used a random 10% [V]. The epoch count was itself chosen by grid search over {1, 10, 50, 100, 1000} [V]. So features see the data once but the readout sees it 50 to 200 times.
- **Results** (SoftHebb / backprop on the same network / random weights) [V]:
  - MNIST 99.35 ± 0.03 / 99.45 ± 0.02.
  - CIFAR-10 80.31 ± 0.14 / 83.97 ± 0.07.
  - STL-10 76.23 ± 0.19 / 74.51 ± 0.36 / 68.2.
  - ImageNet 27.3 (one seed) / not run / 14.0.
  - Hard winner-take-all on STL-10: 54.8. Single layer on CIFAR-10: soft 71.10, hard 62.69, random 63.93.
- **No equal-exposure comparison.** Backprop's epoch count is not stated; there is no one-epoch backprop row [V].
- **Statistics.** Four runs per experiment, one for ImageNet [V].
- **Baselines.** A table of 11 published local or bio-plausible methods on CIFAR-10 (feedback alignment 71.8, direct feedback alignment 70.5, equilibrium propagation 78.5, Burstprop 79.9, hard winner-take-all 72.2 and 64.6, and others) and a second table on STL-10 (CLAPP 73.6, K-means 74.1, Greedy InfoMax 81.9, SimCLR 86.3) [V].
- **Ablations.** Activation function (80.31 with the chosen one, 54.05 to 79.08 otherwise), temperature, width, initial weight radius, learning-rate schedule [V].
- **Length.** 28 pages on arXiv; 5 main sections, 2 appendices, 5 main figures and 2 main tables [V].
- **Admitted limits.** "we have tested SoftHebb only in computer vision tasks"; "it is unclear how to apply SoftHebb to generic deep network architectures"; "SoftHebb cannot compete with the true state of the art in ML" [V].
- **Peer review.** ICLR 2023, notable top 25%, oral [V] ([ICLR page](https://iclr.cc/virtual/2023/oral/12759)). Reviewer recommendations 6, 6, 6, 8 with confidences 3, 3, 4, 4, mean 6.5, taken from a public secondary dataset of ICLR scores [V]. The review text itself could not be fetched (OpenReview blocked automated access).
- **Follow-ups that matter here.**
  - The earlier single-layer paper (Moraitis et al., *Neuromorphic Computing and Engineering* 2022, [arXiv 2107.05747](https://arxiv.org/abs/2107.05747)) contains the only explicit one-pass win: in "a truly single-epoch scenario … Strikingly, it even outperforms end-to-end (e2e) BP in 1-epoch accuracy", with SoftHebb at batch size 1 and backprop tuned to batch size 4 [V]. MNIST only, 2,000 neurons; the one-epoch numbers are in a figure. At 100 epochs backprop wins, 98.65 ± 0.06 against 97.80 ± 0.02 [V].
  - SPHeRe (arXiv 2510.14810): CIFAR-10 81.11; Split-CIFAR100 72.72 against a re-implemented SoftHebb at 51.1; criticises SoftHebb as "sensitive to network structure" [V].
  - Self-Contrastive Forward-Forward (*Nature Communications* 2025): CIFAR-10 80.8, STL-10 77.3 on the same architecture [V].
  - "Hebbian Learning with Global Direction" (ICASSP 2026, arXiv 2601.21367): adds the sign of the backprop gradient; ImageNet 53.71 against SoftHebb 26.08 and backprop 58.84 [V].
  - Incremental audio classification with SoftHebb (ICASSP 2026, arXiv 2604.18270): 76.3% against 68.7% on a five-step task-incremental ESC-50, with a backprop-trained head per task [V].
  - No paper was found that attaches language or vision-and-language learning to SoftHebb-style features.

### 1.9 Monty, "Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference" — [arXiv 2507.04494](https://arxiv.org/abs/2507.04494)

- **Claims.** "the first quantitative demonstration of Monty's capabilities"; "associative, Hebbian-like binding to enable rapid, continual, and computationally efficient learning, properties that compare favorably to current deep learning architectures" [V]. Hedge: "The primary purpose is not to argue that Monty, in its current form, is superior to all learning systems" [V].
- **Learning in fact.** Stores points (location, pose, features) in a per-object 3D reference frame; "the simplest possible form of Hebbian learning" [V]. It is supervised: "Monty is provided with the ground-truth values of the object ID and orientation" [V]. Memory grows with every object [V].
- **Data.** 77 YCB objects in the Habitat simulator, simulated colour-and-depth patches, 14 training rotations per object. No real images or sensors [V].
- **Results.**
  - Recognition 98.6% in the base condition, 88.1% with noise and novel rotations, 73.1% with colour removed as well [V].
  - Few-shot: after one view about 50% against about 30% for a from-scratch Vision Transformer (chance 1.3%); after eight views 88% [V]. A Vision Transformer trained for one epoch "never achieves accuracy significantly above chance" [V]. A pretrained one "achieves similar learning efficiency to Monty" on classification [V].
  - Continual learning over 77 one-object tasks: "only a minor degradation" against "catastrophic forgetting", with no numbers in the text and only naive fine-tuning as baseline [V].
  - Training compute about 528,000,000 times less than the pretrained transformer once pretraining is counted, about 34,000 times less than from scratch [V, D].
- **Statistics.** No seed count, error bars or tests in the text [V].
- **Length.** arXiv "32 pages, 8 figures"; about 16,000 words; journal version 52 pages [V].
- **Admitted limits.** "we have limited our evaluations to the context of 3D object perception"; unsupervised learning "was beyond the scope"; "continuously expands its representational capacity when learning"; inference cost "scales linearly with the number of known models" [V].
- **Status.** *Neural Computation* 38(6):845–896, DOI 10.1162/NECO.a.1508, issue dated 2026-05-20 [V] ([Crossref](https://api.crossref.org/works/10.1162/NECO.a.1508)).
- **Reception.** No public reviews or critiques found; Hacker News has no thread on the paper [V]. Code repository 612 stars [V]. One independent follow-up (arXiv 2605.22206, a spiking reinterpretation, synthetic data only) [V]. The companion white paper (arXiv 2412.18354) is still a preprint; the theory paper appeared in *Neural Computation* 38(11), Oct 2026 [V].

---

## 2. Work the first survey missed (2022–2026)

Checked against the first survey by keyword; items already covered there are not repeated.

### (a) Local-rule systems that learn word–object mappings

**No paper was found, 2022–2026, in which a Hebbian or other non-backprop learner names real photographs or video frames.** Queries on Hebbian, spiking, self-organising-map and associative word learning with real, egocentric or headcam images returned only backprop-trained models or symbolic-input Hebbian models. The nearest items:

- **Tomasello, Garagnani, Wennekers and Pulvermüller 2018**, *Frontiers in Computational Neuroscience* 12: spiking, Hebbian, 12 cortical areas with anatomical connectivity; learns object and action words from paired patterns [V as cited in 2507.11788].
- **Constant, Pulvermüller and Tomasello 2023**, *Cerebral Cortex* 33(11), "Brain-constrained neural modeling explains fast mapping of words to meaning" — [open full text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10233283/). Read in full text this session.
  - 12 areas of 625 excitatory cells each; Hebbian learning with potentiation and depression; local inhibition; noise [V].
  - Six words per network (three object, three action); each input is "22 randomly selected cells from the 625" of a primary area. Random patterns, not images [V].
  - Two-stage learning: a first stage forms word-form circuits (articulatory plus auditory, likened to babbling) and referent circuits separately; a second stage pairs them [V].
  - Result: with the first stage, "a mean of 26.67% (SE = 4.59) of the symbolic links per network being made after just 5 presentations"; without it, links "did not start emerging until after 30 learning events and only reached a mean of 53.85% (SE = 2.77) … after 100 presentations" [V]. Abstract: 1–10 events against 40–100 [V].
  - 22 networks, 2 removed for merged representations, 20 analysed; Poisson and negative-binomial regression, Bonferroni correction [V].
  - Admitted limits: word forms may not exist whole before meaning; "replicate the results with thousands of words in a much larger network" is future work [V].
  - 26 citations (OpenAlex) [V].
- **Follow-ups in that line** [S]: Henningsen-Schomers, Garagnani and Pulvermüller on concrete and abstract concepts (2022–2023); Nguyen, Henningsen-Schomers and Pulvermüller 2024 on category terms and proper names; a 2026 *Frontiers in Human Neuroscience* model with tractography-derived connectivity. None was read.
- **WOLVES** (Bhat, Spencer and Samuelson, *Psychological Review* 129(4):640–695, 2022) [S]: a dynamic-neural-field model of cross-situational word learning that joins looking and word–object binding; laboratory stimuli, memory traces not backprop. Extended to infancy in *Child Development* 2023 [S].
- **Neuromorphic interactive object learning** (Hajizada et al., ICONS 2022, best paper [S]) and its successor **CLP-SNN** on Loihi 2 ([arXiv 2511.01553](https://arxiv.org/abs/2511.01553), v3 Oct 2026): a local self-normalising rule, new prototype neurons recruited on novelty, reward-gated plasticity; on OpenLORIS it "matches Replay and SLDA … at 1-shot" and at 25-shot is within 0.8 points of replay and 6.2 below streaming discriminant analysis [V]. Works on features from an extractor whose training is not stated on the abstract page. Object instances, not words.
- **Self-organising map with raw object images and phoneme streams** (arXiv 1905.08300, 2019) [S]: outside the window but the only self-organising-map word learner on raw images that surfaced; up to 78% word–referent accuracy in ambiguous trials [S].

### (b) Non-backprop models with a developmental curriculum

- **Constant et al. 2023** (above) is the clearest case: a babbling-like stage first, word–meaning pairing second, with the speed-up measured.
- **Class-incremental curriculum for BrainProp** (van den Berg, Roelfsema and Bohte, IJCNN 2025) [S]: a local reinforcement-style rule with classes introduced progressively on a performance criterion; image classification, no language.
- **Staggered critical periods with a local rule** (Delrocq et al., CCN 2025 abstract) [S]: visual areas trained in sequence. Already named in the first survey in another role.
- **Developmental grow-then-prune spiking networks** and **developmental pruning** (arXiv 2211.12714) [S]: structure changes during learning; no language.
- **Backprop contrast**: Sheybani et al., "Curriculum Learning with Infant Egocentric Videos", NeurIPS 2023 spotlight [S]: ordering infant headcam video by age helps self-supervised models; youngest-first is best.
- **Nothing found** that combines staged language development (babbling, echo, first words, two words) with a non-backprop learner on real data.

### (c) Slow learner plus episodic one-shot store for word learning

- **Hill et al., "Grounded Language Learning Fast and Slow"** ([arXiv 2009.01719](https://arxiv.org/abs/2009.01719); ICLR 2021 [R]): an agent with "a novel dual-coding external memory" shows "one-shot word learning when trained with conventional RL algorithms"; after "This is a dax" it can "Put the dax on the bed"; generalises within ShapeNet category "under certain training conditions" [V abstract]. Backprop, simulated 3D world. It is the standard citation for "fast mapping = episodic memory plus slow weights".
- **Constant et al. 2023** gives the opposite account: fast mapping without any separate store, from pre-formed circuits and hub areas [V].
- **Hosoda et al. 2024**, *Frontiers in Neuroscience*, "A single fast Hebbian-like process enabling one-shot class addition in deep neural networks without backbone modification" [S]: adds a class to a pretrained ImageNet classifier from one image by a Hebbian weight imprint, at about 80% of original accuracy [S]. Backbone is backprop-trained.
- **"Tuning-less Object Naming with a Foundation Model"** ([arXiv 2311.04924](https://arxiv.org/abs/2311.04924), DISA 2023): one-shot naming on a robot by storing frozen foundation-model features against a growing vocabulary; no accuracy figure in the abstract [V].
- **Theory, no simulation**: Davis and Gaskell's complementary-systems account of word learning and Gaskell's 2024 review [S].
- **Nothing found**: a neural or Hebbian slow learner with an episodic store tested on fast mapping of names for real images.

### (d) Equal-exposure, single-pass or data-efficiency comparisons

- **BabyCL** (Jiang, Yang, Norman, Lake and Ren, [arXiv 2606.05115](https://arxiv.org/abs/2606.05115), 3 Jun 2026; NeurIPS 2026 BabyVLM workshop [V]). Read in the main parts this session.
  - One chronological pass over SAYCam child S: 221 hours, 37,486 utterances, 600,285 frames [V].
  - Four-way forced choice on Labeled-S (chance 25%), mean with approximate 95% interval across seeds: offline contrastive model at 400 epochs 57.31 ± 1.13; naive one-pass 27.52 ± 3.12; replay-matched 36.41 ± 2.31; BabyCL 43.38 ± 1.47 [V, Table 1]. The running text gives slightly different values (25.68, 35.32, 44.46, 57.45) [V].
  - Seeds: three in the ablation table [V].
  - Method: event segmentation, two replay buffers holding by default 10% plus 50% of the stream, eight extra passes per insertion (about six revisits per item) [V].
  - Motivation: a critic "argues that this data cycling invalidates any cognitively-meaningful conclusions drawn from these models" (Bowers, *Psychological Review* 2025) [V].
  - Other evaluations used: a Visual Two-Word Test (offline 63.08%, BabyCL 61.01%, one-pass 52.22%, chance 50%) and Baby Winoground [V].
  - Admitted limits: gap to offline remains; "The replay buffers used here are large relative to the size of the video stream" [V].
- **Moraitis et al. 2022** (section 1.8): Hebbian beats tuned backprop within the first epoch on MNIST [V].
- **Song et al., *Nature Neuroscience* 2024** (already in the first survey): predictive-coding networks ahead of backprop in online, small-data and continual settings [S].
- **Lagani et al., *Neural Networks* 2021** ([arXiv 2103.09002](https://arxiv.org/abs/2103.09002)): Hebbian pre-training plus a gradient-trained head "outperforms the other approaches in almost all the cases" when labels are scarce [V abstract, via sub-reader].
- **Monty** (section 1.9): one-view and one-epoch comparisons against a Vision Transformer [V].
- **CLP-SNN** (above): parity with replay and streaming discriminant analysis at one shot [V].
- **Hyperdimensional alignment of frozen models** ([arXiv 2602.23588](https://arxiv.org/abs/2602.23588), Feb 2026): image captioning built "in a single pass over the data" by binding and bundling, no gradient updates, on frozen pretrained encoders; claims performance "comparable to end-to-end vision–language training methods" [V abstract].
- **Counter-evidence**: "Alignment Without Binding" (below) and the first survey's "Price of Locality" both show strictly local credit failing to transfer.

### (e) Backprop-free vision-and-language learning

- **"Alignment Without Binding: A Credit-Depth Threshold for Cross-Modal Transfer in Local Learning"** (Yeh and Sang; NeurIPS 2026 BabyVLM and DevAI workshops [V]; [workshop page](https://neurips.cc/virtual/2026/173594)).
  - Image and caption encoders trained from scratch with a contrastive coupling; the only knob is how many blocks a local gradient reaches (1 = strictly local, 4 = full backprop) [V abstract].
  - About 20,000 image–caption pairs, about 208k words [V].
  - At 156M parameters, strictly local credit reaches 0.99 training retrieval (above backprop's 0.955) yet transfers at only 1.10 times an 8.1% base rate on held-out category precision at 10; one more block of reach recovers most of backprop's lift [V].
  - "a frozen random-initialization encoder transfers better than one trained under strictly local credit", three seeds [V].
  - Caption density is the one intervention that helped, lift 2.26 to 4.35 times [V].
  - Still gradient-based within each block. It is the closest thing to a negative result aimed directly at local multimodal learning, in the venue this work would most naturally go to.
- **Limbacher, Özdenizci and Legenstein** (arXiv 2205.11276, 2022) [S]: spiking network with Hebbian memory associates spoken digits with handwritten digits and regenerates the paired image.
- **Multimodal self-organising maps with Hebbian cross-links** (arXiv 2004.05488, 2020) [S]: written and spoken digits.
- **Constrained competitive Hebbian learning on frozen audiovisual embeddings** ([arXiv 2607.16027](https://arxiv.org/abs/2607.16027), Jul 2026): three audiovisual classification benchmarks; Hebbian gives lower information cost than sparse backprop "rather than uniform accuracy gains" [V abstract]. Classification, not naming.
- **Vision-only local-rule systems, 2026**: a VisNet-style hierarchy of local plasticity ([arXiv 2604.09734](https://arxiv.org/abs/2604.09734)) reaches 80.1% on CIFAR-10 by linear probe after 300 epochs, 78.3% with a nearest-class-mean classifier and no gradients at all, 61.4% with no learning [V abstract]. Brain-like BCPNN with Hebbian and structural plasticity ([arXiv 2406.04733](https://arxiv.org/abs/2406.04733)): MNIST to CIFAR-10 and non-image data, "competitive" with multi-layer perceptrons by linear probe [V abstract summary].
- **Nothing found**: any published system that learns vision-and-language association from real images with no gradients anywhere and reports held-out naming.

---

## 3. What a reviewer will expect

### 3.1 Comparisons

| Expectation | Set by | Why it applies here |
|---|---|---|
| A backprop model of the same shape, **with its learning rate and batch size tuned for the one-pass setting** | Moraitis 2022 tuned backprop to batch size 4 for its one-epoch claim [V] | The 0.52 against 0.39 claim stands or falls on whether the backprop baseline was tuned for one pass |
| A backprop model **with a replay buffer** at matched update count | BabyCL's replay-matched baseline [V] | BabyCL shows naive one-pass backprop is a weak baseline that replay partly repairs |
| The same backprop model trained to convergence, as an upper bound | BabyCL's offline row; SoftHebb's backprop row [V] | Already present (0.50 after five passes); state where it plateaus |
| Closed-form or gradient-free readouts on the same features: nearest class mean, streaming discriminant analysis, ridge regression | CLP-SNN compares against streaming discriminant analysis and replay; arXiv 2604.09734 reports nearest-class-mean [V] | Separates "the features are good" from "the word-cell learning rule is good" |
| Random, untrained features with the same readout | SoftHebb (STL-10 68.2, ImageNet 14.0); "Alignment Without Binding" found random beats strictly local [V] | Winner-take-all filters must beat random filters or the cortex adds nothing |
| Hard against soft competition | SoftHebb (54.8 against 76.2 on STL-10) [V] | The visual cortex here is hard winner-take-all |
| A table of published local-rule numbers | SoftHebb's two tables [V] | Reviewers scored SoftHebb 6, 6, 6, 8 with that table present |
| Ablation of each component | SoftHebb activation ablations; Monty policy on/off; Dragon Hatchling criticised for lacking them [V] | Episodic store on/off is already there (0.9 against 0.1); do the same for dendrites, decorrelation, growth and pruning, reward chooser |
| Continual-learning baselines beyond naive fine-tuning | CMP used online EWC; Bricken used EWC, MAS, SI and the fruit-fly model; Monty used only naive fine-tuning and has no numbers [V] | Forgetting is this system's open failure; report it with the standard backward-transfer measure |
| Learning from pixels, not from pretrained embeddings | Bricken's reviewer: "imagenet embeddings already separate CIFAR10 classes... is this an interesting learning setting?"; area chair held the score down because it did not train "the full network" [V] | This system learns its own visual features, which is the thing that paper was marked down for lacking; say so |
| Few hyperparameters, or an account of how they were set | Bricken's reviewer: "the amount of tuning hyperparameters... is enormous"; SoftHebb grid-searched on labelled validation accuracy [V] | A system with many organs invites the same remark |

### 3.2 Evaluations that count as standard

- **Four-way forced choice on Labeled-S**, 22 categories, chance 25%; the child's-view contrastive model scores 61.6% [V, as restated in arXiv 2507.14749]. Used unchanged by BabyCL [V].
- **Visual Two-Word Test and Baby Winoground** for two-word composition, from the BabyVLM benchmark [V via BabyCL]. This is the nearest published test to the "two words" stage here.
- **Sentences or events to criterion** as a function of lexicon size (assembly-calculus papers, Constant et al.) [V]. Cheap to add: plot exposures per word against vocabulary size.
- **Fast mapping measured as learning events to a reliable link**, with and without the prior stage (Constant et al.: 1–10 against 40–100) [V]. The 0.9 against 0.1 episodic result is the same kind of contrast.
- **Backward and forward transfer** over a task sequence (CMP, after Lopez-Paz and Ranzato) [V].
- **Linear probe after one unsupervised epoch** on CIFAR-10 and STL-10 for any Hebbian visual front end (SoftHebb and its follow-ups) [V].
- **Seeds and intervals.** The range in this set is one to five seeds; BabyCL reports mean with 95% interval over three [V]. Three seeds with intervals matches the field; five would exceed most of it.

### 3.3 Claims that drew fire

- Calling a model Hebbian or brain-like when gradient descent trains it (Dragon Hatchling) [V].
- One baseline, and an old one (Dragon Hatchling) [V].
- Low forgetting reported without the accuracy it costs. CMP avoided this by stating the gap in its abstract [V].
- A custom benchmark with no standard anchor. CMP added Split-MNIST and reported the null [V].
- Training-data cycling presented as child-like (the criticism of the child's-view model that BabyCL answers) [V]. This one favours a single-pass system.

---

## 4. The papers a manuscript must cite and position against

1. **Mitropolsky and Papadimitriou, arXiv 2306.15364 and 2507.11788.** The only other backprop-free word learners with a developmental story; they learn from pre-formed symbolic assemblies in a toy language with no held-out test or baseline, where this system learns from photographs and adult text and is scored on held-out naming against a backprop twin.
2. **Constant, Pulvermüller and Tomasello, *Cerebral Cortex* 2023 (with Tomasello et al. 2018).** Hebbian fast mapping after a babbling-like stage, on six words encoded as random patterns; this system makes the same developmental claim on real images, gets one-shot naming from a separate episodic store instead of pre-formed circuits, and lacks their anatomical realism.
3. **Vong, Wang, Orhan and Lake, *Science* 2024, and Jiang et al., arXiv 2606.05115 (BabyCL).** The backprop standard for word learning from real child-view data and its single-pass version; they need hundreds of epochs or a replay buffer of 60% of the stream, where this system uses no gradients and one pass, but on curated photographs and 16–25 names instead of a child's headcam and 22 categories in four-way choice.
4. **Mayor and Plunkett, *Psychological Review* 2010.** Self-organising maps joined by Hebbian links, the classic non-backprop account of fast mapping and taxonomic generalisation; it used hand-built feature vectors and reported a new-word bias through generalisation, where this system uses real photographs and shows no such bias.
5. **Journé et al., ICLR 2023 (SoftHebb), with Moraitis et al. 2022.** The reference winner-take-all Hebbian visual learner and the only explicit one-epoch win over backprop; its readout is a gradient-trained supervised classifier run for 50–200 epochs and it has no language, where here the readout is also local and trained once.
6. **Shen, Dasgupta and Navlakha, *Neural Computation* 2023 (fruit-fly model), with Bricken et al., ICLR 2023.** The fruit-fly model is a single-pass, local-rule learner that resists forgetting on class-incremental benchmarks, and Bricken's is its backprop-trained cousin; both sit on features from a backprop-pretrained network and both avoid forgetting by freezing all but the target class, where this system learns its own features, shares their sparse competitive coding, and still forgets when the lesson mix changes.
7. **Singh, Norman and Schapiro, PNAS 2022 (with Hill et al., ICLR 2021).** A fast episodic system teaching a slow one through replay, and an episodic memory giving one-shot word learning in a backprop agent; here the episodic store gives the one-shot effect (0.9 against 0.1) but replay from it did not prevent forgetting, which is a negative result against that account.
8. **Leadholm et al., *Neural Computation* 2026 (Monty).** The most visible published non-deep-learning system claiming rapid and continual learning against a backprop baseline; it is supervised storage on 77 simulated objects with no language and no reported variance, where this system is real photographs, words, and seeds.

Optional ninth, as the cautionary citation: **Yeh and Sang, "Alignment Without Binding" (NeurIPS 2026 workshops)**, which reports that strictly local credit fits image–caption training data yet barely transfers; this system's held-out naming at 0.52 against chance 0.05 is a counter-example worth stating in those terms, with the caveat that the settings differ.

---

## 5. Gaps and things not verified

- **OpenReview could not be fetched directly** (automated access blocked). Sparse Distributed Memory reviews, scores and the area chair's note were read from public mirrors of the review data. For SoftHebb only the scores were obtained, from a secondary public dataset; its review text and meta-review were not.- **Google Scholar counts** were not obtainable; citation counts are Semantic Scholar and OpenAlex only, and the two disagree by a few on small counts.
- **Figures were not read.** One-epoch accuracies in Moraitis 2022, the Dragon Hatchling scaling curve and Monty's continual-learning curves exist only as plots.
- **Not read in full text**: WOLVES, Hill et al. beyond the abstract, Hosoda et al., the Berlin group's 2024–2026 follow-ups, "Alignment Without Binding" beyond its abstract (no arXiv copy found), Bowers 2025.
- **The DevAI workshop's accepted list** was not found; only the BabyVLM list (12 papers) was read, and it contains two relevant papers (BabyCL and "Alignment Without Binding") [V].
- **A negative search is not proof of absence.** The statement that nobody has published gradient-free naming of real photographs rests on about 25 queries across the five categories.
- **Inconsistencies found inside sources**, to be aware of when citing: BabyCL's table and text disagree by about one to two points; CMP repeats one figure across two different experiments; the Dragon Hatchling abstract says 1B parameters and the tables stop at 800M.
