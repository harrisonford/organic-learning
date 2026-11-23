# References to open by hand before anything is sent

For Kris. arXiv has announced bans for invented references, so each entry below must be opened and its title,
authors, year and the claim we attach to it confirmed. Tick the box when done; strike the entry if it does not say
what we say it says. **Status** is how the assistant knows the entry:

- **read**: the paper's own text was read in this project (research_notes/, or fetched for the probe)
- **abstract**: only the abstract or a listing was seen
- **memory**: written from the assistant's memory; the bibliographic details may be wrong

| ✓ | Reference | Link | What the draft uses it for | Status |
|---|---|---|---|---|
| ☐ | Atick & Redlich (1992). What does the retina know about natural scenes? *Neural Computation* 4(2), 196–210 | https://doi.org/10.1162/neco.1992.4.2.196 | the retina decorrelates natural scenes | memory |
| ☐ | Coates, Lee & Ng (2011). An analysis of single-layer networks in unsupervised feature learning. *AISTATS*, PMLR 15, 215–223 | https://proceedings.mlr.press/v15/coates11a.html | the "triangle" response; whitening; random patches are hard to beat | memory |
| ☐ | Saxe, Koh, Chen, Bhand, Suresh & Ng (2011). On random weights and unsupervised feature learning. *ICML* | https://icml.cc/2011/papers/551_icmlpaper.pdf | random filters with pooling do nearly as well as learned ones | memory |
| ☐ | Földiák (1990). Forming sparse representations by local anti-Hebbian learning. *Biological Cybernetics* 64, 165–170 | https://doi.org/10.1007/BF02331346 | learned lateral inhibition decorrelates | memory |
| ☐ | Zylberberg, Murphy & DeWeese (2011). A sparse coding model with synaptically local plasticity and spiking neurons can account for the diverse shapes of V1 simple cell receptive fields. *PLoS Computational Biology* 7(10), e1002250 | https://doi.org/10.1371/journal.pcbi.1002250 | local Hebbian/anti-Hebbian rules give V1-like fields | read (text, not the equations) |
| ☐ | Pehlevan & Chklovskii (2014). A Hebbian/anti-Hebbian network derived from online non-negative matrix factorization can cluster and discover sparse features. *Asilomar Conference on Signals, Systems and Computers* | https://arxiv.org/abs/1503.00680 | the lateral network's rules (their equations 7 and 8), used unchanged | read |
| ☐ | Urbanczik & Senn (2014). Learning by the dendritic prediction of somatic spiking. *Neuron* 81(3), 521–528 | https://doi.org/10.1016/j.neuron.2013.11.030 | the word cells' dendritic rule | memory |
| ☐ | Rescorla & Wagner (1972). A theory of Pavlovian conditioning. In Black & Prokasy (eds.), *Classical Conditioning II*, 64–99 | (book chapter) | cue competition in the word cells | memory |
| ☐ | McClelland, McNaughton & O'Reilly (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review* 102(3), 419–457 | https://doi.org/10.1037/0033-295X.102.3.419 | fast and slow memory | memory |
| ☐ | Krotov & Hopfield (2016). Dense associative memory for pattern recognition. *NeurIPS* | https://arxiv.org/abs/1606.01164 | the store's steep comparison | memory |
| ☐ | Krizhevsky (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto | https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf | CIFAR-10 and CIFAR-100 | memory |
| ☐ | Hayes & Kanan (2020). Lifelong machine learning with deep streaming linear discriminant analysis. *CVPR Workshops* | https://arxiv.org/abs/1909.01520 | streaming LDA; it sits on pretrained features | abstract |
| ☐ | Shen, Dasgupta & Navlakha (2021/2023). Algorithmic insights on continual learning from fruit flies; journal version: Reducing catastrophic forgetting with associative learning: a lesson from fruit flies. *Neural Computation* | https://arxiv.org/abs/2107.07617 | a local-rule continual learner on pretrained features | read (journal title confirmed) |
| ☐ | Bricken, Davies, Singh, Krotov & Kreiman (2023). Sparse distributed memory is a continual learner. *ICLR* | https://arxiv.org/abs/2303.11934 | trained by backpropagation on pretrained embeddings | read |
| ☐ | "The Art of Not Forgetting: A Local Learning Architecture for Continual Learning" (2026), arXiv 2607.26523 | https://arxiv.org/abs/2607.26523 | a delta rule on a linear readout over fixed random sparse codes | read (authors to be filled in from the page) |
| ☐ | Journé, Rodriguez, Guo & Moraitis (2023). Hebbian deep learning without feedback. *ICLR* | https://arxiv.org/abs/2209.11883 | SoftHebb: many epochs, a gradient-trained head | read |
| ☐ | Moraitis, Toichkin, Journé, Chua & Guo (2022). SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks. *Neuromorphic Computing and Engineering* | https://arxiv.org/abs/2107.05747 | a single-epoch advantage over backpropagation | abstract |
| ☐ | Mitropolsky & Papadimitriou (2023). The architecture of a biologically plausible language organ | https://arxiv.org/abs/2306.15364 | Hebbian language model, toy input | read |
| ☐ | Mitropolsky & Papadimitriou (2025). Simulated language acquisition in a biologically realistic model of the brain | https://arxiv.org/abs/2507.11788 | the same, 2- and 3-word sentences | read |
| ☐ | Tomasello, Garagnani, Wennekers & Pulvermüller (2018). A neurobiologically constrained cortex model of semantic grounding with spiking neurons and brain-like connectivity. *Frontiers in Computational Neuroscience* 12, 88 | https://doi.org/10.3389/fncom.2018.00088 | brain-constrained Hebbian word learning | memory |
| ☐ | Constant, Pulvermüller & Tomasello (2023). Brain-constrained neural modeling explains fast mapping of words to meaning. *Cerebral Cortex* 33(11), 6872–6890 | https://doi.org/10.1093/cercor/bhad007 | fast mapping after a babbling-like stage | abstract |
| ☐ | Jiang, Yang, Norman, Lake & Ren (2026). BabyCL, arXiv 2606.05115 (title to be filled in from the page) | https://arxiv.org/abs/2606.05115 | one chronological pass is near chance without large replay buffers | read |
| ☐ | Hill, Tieleman, von Glehn, Wong, Merzic & Clark (2021). Grounded language learning fast and slow. *ICLR* | https://arxiv.org/abs/2009.01719 | one-shot word learning with an episodic memory, by gradient | abstract |
| ☐ | Carey & Bartlett (1978). Acquiring a single new word. *Papers and Reports on Child Language Development* 15, 17–29 | (no DOI) | fast mapping in children | memory |
| ☐ | Horst & Samuelson (2008). Fast mapping but poor retention by 24-month-old infants. *Infancy* 13(2), 128–157 | https://doi.org/10.1080/15250000701795598 | a fast-mapped word is soon lost | memory |
| ☐ | Okanda & Itakura (2008): the yes bias of two-year-olds (exact paper to be identified; they have several) | (to find) | two-year-olds say "yes" to yes/no questions | memory, uncertain |
| ☐ | Rolls (1999). Spatial view cells and the representation of place in the primate hippocampus. *Hippocampus* 9(4), 467–480 | https://doi.org/10.1002/(SICI)1098-1063(1999)9:4%3C467::AID-HIPO13%3E3.0.CO;2-F | hippocampal cells tied to where the eyes look | memory |
| ☐ | Liu, Shen, Olsen & Ryan (2017). Visual sampling predicts hippocampal activity. *Journal of Neuroscience* 37(3), 599–609 | https://doi.org/10.1523/JNEUROSCI.2610-16.2016 | more fixations at study, more hippocampal activity | memory |

Not cited yet but in the notes, if wanted: the Dragon Hatchling (arXiv 2509.26507; trained by backpropagation
through time), Monty (arXiv 2507.04494), "Alignment Without Binding" (workshop, 2026; a frozen random encoder beat
local credit assignment).
