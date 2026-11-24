# How brains learn where to look: a survey, 2026-11-23

Asked for by Kris ("lets improve the gaze"). Made by a search agent with generic queries; about 45 sources opened
(some as abstracts only, through Europe PMC). Not yet opened by a person. One gap it reports: no model of
2018–2026 learns a whole scene-viewing priority map with purely local rules; the models below are pieces.

## What is wiring and what is learned

**Wiring.** A priority map whose peak is the next fixation (LIP, frontal eye fields, superior colliculus; Bisley &
Goldberg 2010; Fecteau & Munoz 2006: priority = salience and relevance together). The basal ganglia gate: the
substantia nigra holds the colliculus shut, the caudate opens it for a place (Hikosaka, Kim, Yasuda & Yamamoto
2014). Winner-take-all; inhibition of return. The newborn's looking is this reflex: drawn to contrast, edges,
movement and faces, under subcortical control (Oakes 2023), and "sticky": only at 4 months do infants let go of
what they fixate with ease, anticipate, and learn that a cue predicts a place (Johnson, Posner & Rothbart 1991;
Haith, Hazan & Goodman 1988).

**Learned by reward.** Which things open the gate. With unequal reward the caudate's responses change until its
preferred direction can flip, and saccades to the rewarded side come earlier (Kawagoe, Takikawa & Hikosaka 1998).
Two memories of value: a flexible one (caudate head; learned in a few trials, reversed in a few) and a stable one
(caudate tail; built over days for hundreds of objects, kept for months, driving gaze with no reward in sight)
(Kim & Hikosaka 2013; Hikosaka et al. 2019). The teacher is dopamine's reward prediction error (Schultz, Dayan &
Montague 1997). The rule is three-factor: coincidence of input and output leaves a trace on the synapse, and
dopamine arriving within about 0.3–2 s turns the trace into a change (Yagishita et al. 2014; Gerstner et al.
2018). A thing once paired with reward goes on capturing attention when it no longer pays (Anderson, Laurent &
Yantis 2011). Infants of 6–8 months learn within about three trials to make the eye movement that makes
something happen (Wang et al. 2012).

**Learned from words.** A heard word draws the eye to what it names, in adults as the word unfolds (Tanenhaus et
al. 1995; Huettig, Rommers & Meyer 2011) and in infants from 6–9 months (Bergelson & Swingley 2012); the time to
shift falls from 984 ms at 15 months to 769 ms at 25 (Fernald, Perfors & Marchman 2006). The account: Hebbian
links between words and things, and competition among the things in view (McMurray, Horst & Samuelson 2012). In
the brain a template of the wanted features is held in prefrontal cortex and biases the map: with it switched
off, selection by feature is lost and selection by place is spared (Bichot et al. 2015).

**Looking to find out.** People fixate where a look will most reduce uncertainty about the target, and are close
to the ideal searcher (Najemnik & Geisler 2005). LIP fires more before a saccade that will settle an uncertain
decision, apart from reward (Horan, Daddaoua & Gottlieb 2019); dopamine neurons that code reward also code the
promise of advance information (Bromberg-Martin & Hikosaka 2009). Surprise draws gaze (Itti & Baldi 2009).

**How long to stay.** A rise to a threshold: a saccade happens when the movement activity reaches a fixed level,
and the time it takes varies with how fast it rises (Hanes & Schall 1996; Roitman & Shadlen 2002); in scenes a
timer slowed by processing difficulty, mean fixations 230–270 ms (Nuthmann et al. 2010).

**The fovea.** Cortical magnification M = M0 / (1 + E/E2), E2 about 0.6–1°: detail about 2 times coarser at 3°,
4–5 times at 10°, 8 times at 20° (Strasburger 2021; Drasdo et al. 2007; Curcio et al. 1990).

**Gaze following.** From 3–4 months with a salient thing near the adult's head, by 12 months to things behind the
infant; learned from the reward of what the adult looks at (Deák et al. 2014; Triesch et al. 2006, a model:
reward for the fixated thing habituates and recovers).

## Models that learn gaze without backprop
Hong & Hikosaka 2011 (frontal eye field, caudate D1/D2, nigra, colliculus: an eligibility trace and dopamine);
N'Guyen, Thurat & Girard 2014 (actor-critic through basal ganglia loops); Brown, Bullock & Grossberg 2004
(TELOS); Vitay & Hamker 2010 (covariance rule gated by dopamine); Triesch et al. 2006 (gaze following by TD);
Rombouts, Bohte & Roelfsema 2015 (AuGMEnT: tags set by feedback from the chosen action, then dopamine);
Rakhshan, Schafer, Moore & Soltani 2025 (reward-dependent Hebbian target selection in the frontal eye field).

## Mechanisms a small learner could have, from the best established down
1. Dopamine-gated plasticity opening a gate on the priority map: e ← e·exp(−dt/τ) + pre·post, τ about 1 s;
   δ = r − V; Δw = η·δ·e.
2. A priority map that sums its inputs and takes the peak: salience (our reflex is this term), the goal, what
   was just looked at, value.
3. Two memories of value, a fast one and one a hundred times slower that does not fade.
4. Staying as a rise to a threshold, slowed by how hard the processing is.
5. A template from the heard word: links from word to seen features, learned while a thing is fixated and
   named; at the question the word's template is laid over the coarse periphery.
6. Habituation of what is fixated (the letting go that infants gain by 4 months).
7. Information gained by a look as part of the teaching signal (the neural signals are established; their use
   as a plasticity signal for gaze is an extrapolation).
8. Tags set by feedback from the chosen action (a model's proposal; least direct evidence).

## How the map combines what stands out with what is sought (second survey, 2026-11-24)

Asked by Kris ("why fixed, research a better solution") after the serial gaze was found to add the two with a
hand-set half. A survey agent read abstracts and some full texts; nothing here was opened by a person yet.
- **No single rule is established at the map itself.** Reviews (Bisley & Mirpour 2019; Fecteau & Munoz 2006) say
  the map is driven by salience and by top-down signals and read out by winner-take-all, probably in the
  intermediate superior colliculus. LIP sums disparate signals (Ipata et al. 2009); Guided Search 6.0 (Wolfe 2021)
  builds the map as a weighted average of sources with no fixed weights.
- **Upstream the goal acts as a gain** on feature-tuned cells (Treue & Martínez-Trujillo 1999; Reynolds & Heeger
  2009: R = A·E / (S + σ), attention multiplying the drive before normalisation).
- **Alternatives compete by divisive normalisation**: LIP for value (Louie, Grattan & Glimcher 2011:
  R = Rmax·(V_in + β) / (σ + V_in + V_out)); superior colliculus and optic tectum (Vokoun et al. 2014; Mysore &
  Knudsen 2011: global divisive inhibition gives a switch-like "strongest against the rest"; Banerjee, Kothari &
  Mysore 2026). A model with target-specific gain and local normalisation predicts human fixations in search, and
  without the normalisation salient items dominate (Miconi, Groomes & Kreiman 2016).
- **Time sets the balance too**: saccades under about 150 ms follow salience, those after about 250 ms the goal
  or value (van Zoest, Donk & Theeuwes 2004; Donk & van Zoest 2008; Siebold et al. 2011; Schütz, Trommershäuser &
  Gegenfurtner 2012). Salience is a transient and the goal arrives later and stays.
- **Priority is learned from reward**: reward training alters priority maps durably (Chelazzi et al. 2014),
  value-driven capture (Anderson et al. 2011; with striatal dopamine, Anderson et al. 2016), the caudate tail to
  the colliculus (Hikosaka et al. 2014); a rule: Δw = β·δ·tag, tags set by pre × post activity gated by feedback
  from the chosen response (Rombouts et al. 2015); the gain of each feature map as its signal-to-noise over the
  maps' mean (Navalpakkam & Itti 2007). A single learned salience-to-goal ratio is *not* an established finding.
- **Infants**: fixations of three-month-olds are best predicted by salience, faces take over by 6–9 months (Frank,
  Vul & Johnson 2009; Kwon et al. 2016); word-driven looking from 6 months (Bergelson & Swingley 2012). What
  changes is thought to be endogenous control and experience; no mechanism is shown.
- No model of 2020–2026 learns a saccade map with strictly local rules (closest: Cox et al. 2022; Leukos, Liang &
  Lindsay 2026; N'Guyen, Thurat & Girard 2014).
**Built from this** (VISION.md, 2026-11-24): each source's places compete (each place's share of the source's whole
activity), the map is the sources' weighted average, and the two weights are learned from reward by the rule the
eye commands already have, on a fixed total, starting from the reflex alone. Established: normalisation across
alternatives; reward reshaping priority. Our reading, not a finding: that the balance is two scalar weights.
Not built: the time course (a transient salience against a sustained goal), which would set the balance by when
a look is made and needs time constants a tick does not resolve.

