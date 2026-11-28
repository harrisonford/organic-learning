# How brains come to wait and answer: vocal turn-taking in infants, marmosets and songbirds

Literature research, 2026-11-28. The question behind it: a simulated infant-like learner babbles vowel-like sounds at a fixed rate of about two per second, forever, whether or not a voice is present. What do real brains do instead, by which mechanisms, and with which numbers?

## How to read this report

- Every reference in the list at the end was opened on PubMed, PMC, Crossref or OpenAlex. Each one is marked:
  - **full text**: the article body was read (PMC text or PMC page).
  - **abstract**: only the abstract was read.
  - **listing only**: only the bibliographic record exists online without an abstract. These are named for orientation and are never the source of a claim; where their content is described, the description comes from an opened paper that is named.
- Numbers are quoted with the paper they come from. "Not found" means the number was not in what could be read.
- Seven papers were read through a page-fetch tool that returns extracts of the PMC page rather than the raw text: Choi 2015, Roy 2011, Borjon 2016, Liao 2018, Miller 2009, Chow 2015, Coleman 2021. Their numbers below should be rechecked against the PDF before they are built into code, the Choi 2015 model constants above all.
- The full text of Takahashi, Narayanan & Ghazanfar 2013 (the coupled-oscillator paper) could not be opened (publisher returned 403; it is not in PMC). Only its abstract was read. What is said here about its analysis comes from later opened papers that describe it (Takahashi 2016, Ghazanfar & Takahashi 2014, Grijseels 2024). **Its model equations and time constants are not verified and are not given here.** The closest published equations that could be read are from the same laboratory (Choi 2015).
- Grades: **(a)** well established, **(b)** supported by a few studies, **(c)** a proposal.

## Short answer

1. No source describes a fixed-rate, endless emission. Awake infants who are alone produce about 4 speech-like sounds per minute (Oller 2019), not 120, and spend roughly 5 to 7 percent of the time vocalising (Iyer 2016). Sounds come in clusters with pauses, and their timing follows arousal.
2. In the first months a voice mostly *elicits* infant vocalising rather than silencing it. About a third of infant vocalisations overlap the mother's from birth to five months (Dominguez 2016, Gratier 2015, Hilbrink 2015). Alternation becomes the dominant pattern gradually, and studies disagree on when.
3. The best-supported mechanism for "stop while the other is calling, answer after" is not a rule but two interacting things: an internal drive to vocalise that builds up and is discharged by calling, and inhibition of that drive (or of the premotor circuit) by the sound of another voice, followed by release. This is causal in zebra finches (Benichov 2020, Norton 2022), recorded in duetting wrens (Coleman 2021), behavioural in marmosets (Roy 2011, Choi 2015), and old knowledge in frogs and insects (Greenfield 1994).
4. The specific claim that marmoset turn-taking is a *coupled oscillator* is contested. A larger replication found call timing to be stochastic, with a refractory period and a response window instead (Grijseels 2024).
5. Waiting is partly learned from contingent replies. Infants learn that vocalising gets answered between 2 and 5 months (Elmlinger 2023); marmoset parents stop answering calls that interrupted them (Chow 2015). Whether parental feedback *speeds up* turn-taking is disputed (Takahashi 2016 found no relation).

---

## 1. Spontaneous vocalisation in infants

### 1.1 How much, alone versus with a caregiver

| Measure | Value | Source |
|---|---|---|
| Speech-like sounds ("protophones") when alone and awake, all-day home recordings | average 4.0 per minute, over 220 five-minute segments | Oller 2019 (full text) |
| Same recordings, was the infant upset? | 60% of alone segments had no crying or whimpering; 93% had at least some protophones | Oller 2019 |
| Time occupied by protophones in the laboratory | more than 4 s per minute with no adult speech and with infant-directed speech (no significant difference between the two); about 3 s per minute when the adults talked to each other | Oller 2019 |
| Percent of time vocalising, 2 to 11 months, laboratory | means 6.53% (parent silent), 6.17% (parent talking to infant), 5.15% (parent talking to interviewer) | Iyer 2016 (full text) |
| Share of protophones directed at no one | about 75% "endogenous", above 50% at all three ages studied, in both the engaged and the independent condition (6 infants) | Long 2020 (full text) |
| Prior work as summarised by Long 2020 | more than 4 protophones per minute over all-day recordings across the first year, more than 3 per minute when alone | Long 2020, citing earlier work not opened here |
| Range of group means at six months across laboratory studies | 1.3 to 11.3 vocalisations per minute | Franklin 2014 (full text), citing two studies not opened here |
| Five-month-olds alone versus with a responding mother | more sound when alone, more still with amplified feedback of their own voice; more repetition of the same phrase when alone | Shimada 2012 (abstract) |
| Parents' infant-directed speech | more than 12 utterances per minute in the laboratory, almost 2 per minute in all-day recordings | Oller 2019 |

By age:

- Vocalisations are present in preterm infants at 32 weeks postmenstrual age, eight weeks before term (Caskey 2011, abstract).
- Protophones outnumber cries by a factor of at least five across the first year in all-day recordings (Oller 2021, full text).
- In laboratory sessions volubility fell with age between 2 and 11 months (Iyer 2016). Oller 2021 suggests the laboratory decline in the second half-year may reflect infants crawling off to explore. A per-month table of rates alone versus with caregiver for 0 to 12 months was **not found** in the text that could be read; the values are in figures.

Plainly: infants vocalise a lot when nobody answers, and the presence or absence of infant-directed talk changes the amount little (Iyer 2016, Oller 2019). Hearing adults talk *to each other* lowers it (Iyer 2016, Oller 2019; at 12 to 13 months infant vocalisations were also less likely after other-directed adult speech at lags up to 5 s, Pretzer 2019, abstract).

Grade: (a) for "infants vocalise spontaneously at a few per minute, mostly undirected". These are observational data from one research group plus one independent study.

### 1.2 Continuous, or in bouts?

In bouts, at several nested timescales.

- A single vocalisation at 2 to 5 months lasts 870 ms on average (SD 662, range 50 to 3640 ms). Pauses between two vocalisations by the same infant average 746 ms (SD 557), within a coding window that by definition only counted pauses of 300 to 3000 ms (Gratier 2015, full text, 51 dyads).
- Across 12 infants the median pause between an infant's own consecutive vocalisations ranged from 307 to 1198 ms at 3 months and from 420 to 1925 ms at 12 months (Hilbrink 2015, Table 5).
- With a mother responding at random, 3-month-olds produced bursts of vocalisations at intervals of 0.5 to 1.5 s, 4-month-olds at 1.0 to 2.0 s. "After vocalizing spontaneously, the infant tended to pause as if to listen for a possible vocal response from the mother. In the absence of a response, he vocalized repeatedly" (Masataka 1993, abstract).
- Syllables sit inside utterances and utterances inside "prelinguistic phrases" with final lengthening (Lynch 1995, abstract; phrase durations in seconds **not found** in the abstract).
- Over a whole day, vocalisation onsets form hierarchically nested clusters across timescales (Abney 2017, full text) and clusters coincide with periods of raised arousal (Wass 2022).
- Inter-vocalisation intervals are shorter right after an adult responds (Ritwika 2020, abstract).
- Within an utterance of canonical babbling the jaw cycles at about 3 per second (Dolata 2008, abstract; MacNeilage 1998). This is the rhythm inside one utterance, not the rate at which utterances are emitted.

Arithmetic for the simulated learner: two sounds per second is 120 per minute, about thirty times the 4.0 per minute measured in infants alone (Oller 2019).

### 1.3 What drives it

- **Arousal, in marmosets** (b). In adult marmosets alone, heart rate rises for about 23 s before a contact call and drops sharply for about 8 s after it; call timing is locked to the phase of a 0.1 Hz autonomic rhythm (the Mayer wave); calls come about once every 10 s (Borjon 2016, abstract and PMC page; 4 animals). In infant marmosets arousal fluctuates at about 0.1 Hz and modulates the amplitude of the roughly 1 Hz respiratory rhythm, which sets which call is produced (Zhang & Ghazanfar 2016, abstract). Arousal is linked to calling "but not inextricably": timing of a partner's call also matters (Liao 2018). Borjon 2016 describes the mechanism as a weak periodic signal plus noise crossing a threshold, and calling as returning arousal toward its resting state.
- **Arousal, in humans** (b). In 12-month-olds at home, clusters of vocalisations occur during elevated infant and caregiver arousal, more tightly in infants than in adults (Wass 2022, abstract). In 24-month-olds vocalisations are most likely at peaks and troughs of heart-rate fluctuation (Borjon et al. 2024, abstract).
- **Respiration** (a). Phonation rides on the breathing cycle; infant breathing during vocalisation is highly variable across the first year (Boliek 1996, abstract). In the marmoset model part of the 2.2 s post-call refractory period is the breath (Grijseels 2024).
- **An intrinsic oscillator for when to vocalise.** For marmosets an approximately 10 s periodicity was reported (Takahashi 2016 citing Takahashi 2013; Liao 2018 found a latency peak near 10 s when alone), but a larger data set found no oscillatory pattern (Grijseels 2024). For human infants **no evidence was found** of a fixed-period generator of utterance onsets; the clustered, heavy-tailed timing above argues against one. Grade: (c) in humans, contested (b) in marmosets.
- **Exploration and self-hearing** (b). Infants alone vocalise more when their own voice is amplified (Shimada 2012). Oller's group reads the undirected majority as vocal play (Long 2020).

---

## 2. Does hearing a voice suppress or delay infant vocalising?

### 2.1 First months: a voice elicits as much as it suppresses

- Newborns, 2 to 4 days, 15 dyads: one third of newborn vocalisations overlapped a maternal one. Of 119 that followed a maternal vocalisation, 68.9% began within 1 s and 26.9% within 50 ms (Dominguez 2016, abstract).
- Three months, 24 dyads at home, 90 minutes each, coded per second: "vocal onsets for both mother and infant were more likely when the other dyad member was vocalizing" (Anderson 1977, abstract). That is facilitation by the partner's voice, not inhibition.
- Two classic descriptions of the same period, both listing only: Stern et al. 1975 (vocalising "in unison and in alternation" as two modes) and Bateson 1975 (one dyad, 1.5 to 3.5 months, alternation). Hilbrink 2015 summarises Bateson's gaps as 1370 ms for the infant and 1430 ms for the mother, including the preceding utterance.
- 8 to 21 weeks, 51 dyads: of infant vocalisations inside turn-taking sequences 31.5% overlapped the mother; of 454 infant turns 44.5% were latched (gap under 50 ms); switching pauses averaged 730 ms (SD 544). No drop in overlap between 2 to 3 and 4 to 5 months (Gratier 2015).
- 3, 4 and 5 months, 12 dyads followed longitudinally: infants produced "just over a third" of their vocalisations in overlap; mothers 14 to 21% (Hilbrink 2015).
- 4, 7 and 10 months, 41 dyads: "infants had a predominant pattern of covocalizing, whereas the mothers had a predominant turn-taking pattern at all ages"; duration of covocalising fell between 4 and 7 months and infants moved from repeated to single vocalisations between maternal turns (Harder 2015, abstract).

### 2.2 When alternation takes over, and the gap durations

| Study | Ages, N | Finding |
|---|---|---|
| Ginsburg & Kilbourne 1988 (abstract) | 3 dyads | Overlap maximal at 7 to 13 weeks; alternation predominant, peaking at 12 to 18 weeks |
| Elias & Broerse 1996 (abstract) | 48 dyads, 3 to 24 months | Covocalisation falls over the first 18 months then rises; predominance of alternation rises linearly with age |
| Jasnow & Feldstein 1986 (abstract) | 29 dyads, 9 months | Alternation exceeds simultaneous vocalising; partners influence each other's switching pauses |
| Beebe et al. 1988 (abstract) | 4 months | Mothers and infants match switching pauses |
| Bornstein et al. 2015 (abstract, full text) | 684 dyads, 11 countries, 5.5 months | Infant vocalisations are contingent on the offset of the mother's vocalising (onset within 2 s of offset) |
| Hilbrink 2015 (full text) | 12 dyads, 3 to 18 months | Overlap falls from just over a third (3 to 5 months) to roughly 20% at 18 months. Infant median gaps across infants: 346 to 903 ms at 3 months, 323 to 1408 ms at 5 months, 543 to 3297 ms at 9 months, 615 to 1872 ms at 12 months, 485 to 1270 ms at 18 months. Mothers stay stable. Infants are fast early and slow down at 9 months |
| Northrup & Iverson 2020 (abstract) | 30 dyads, 3 to 12 months | Vocal coordination present early, no developmental change across the first year |
| Nguyen et al. 2022 (abstract, full text), meta-analysis | 26 studies, 429 infants, 0 to 70 months | Infants take about 1 s to respond; evidence of change with age inconclusive; infant latency tracks adult latency (adult 400 to 1400 ms goes with infant 600 to 1857 ms) |
| Casillas et al. 2016 (abstract) | 1;8 to 3;5 | Children answer questions more slowly than adults for years; simple answers are fast from early on |
| Adults, for comparison | 10 languages | Gaps of the order of 200 ms; mode between 0 and 200 ms (Stivers 2009; Levinson & Torreira 2015) |

Further: at every age from 3 to 18 months the distribution of infant turn onsets differed from what random vocalising would give, so infants are not vocalising at random while mothers fit around them. But infants "do not yet leave longer gaps after their own vocalizations compared to after their mothers' utterances", as adults do (Hilbrink 2015).

Summary for question 2. A voice does not simply shut an infant up. In the first three to five months roughly a third of infant vocalisations start while the mother is vocalising, and partner voice raises the probability of infant onset. Offset-contingent replies exist from birth (Dominguez 2016) and are reliable by 5.5 months (Bornstein 2015). Overlap declines over months to years, with the reported start of the decline ranging from about 3 months (Ginsburg & Kilbourne 1988) to after 5 months (Hilbrink 2015). Typical infant gap after adult offset is 0.3 to 1 s in the first half-year and longer around 9 months.

Grades: overlap of about one third early on, (a), four independent studies. Gradual shift to alternation, (a) as a direction, (b) as to timing. Slowing at 9 months, (b), one longitudinal study, not confirmed by the meta-analysis.

---

## 3. Mechanistic models

### 3.1 The coupled-oscillator account in marmosets

What it claims (Takahashi 2013, abstract; described in Ghazanfar & Takahashi 2014, Takahashi 2016 and Grijseels 2024):

- Adult marmosets out of sight of each other exchange contact ("phee") calls in long sequences, rarely overlapping.
- Each animal behaves as an oscillator: there is "periodic coupling in the timing of utterances across two interacting individuals" and "entrainment, where if the timing of one individual's vocal output quickens or slows, the other follows suit" (Ghazanfar & Takahashi 2014).
- The key analysis is a phase response curve: the interval between two calls of the same animal with a partner call in between (PR), minus that animal's usual interval when undisturbed (T0), correlates positively with the partner's response interval (R). A short R shortens the cycle, a long R lengthens it (Grijseels 2024, methods, replicating the original analysis).
- Periodicity of call production in adults is "approximately 10 s" (Takahashi 2016, citing Takahashi 2013).
- The authors argue this needs no higher cognition (Ghazanfar & Takahashi 2014).

Structure and constants of the 2013 model: **not verified**, full text not accessible.

### 3.2 The drive-inhibition model with published equations (Choi, Takahashi & Ghazanfar 2015)

This is the published form of "an intrinsic drive that builds up, is inhibited by hearing a conspecific and then recovers". Read through the fetch tool; recheck before use.

- Three nodes: drive D, auditory A, motor M.
- Spontaneous: D′(t) = α|D(t)| − A(t) + e(t). e is zero-mean Gaussian noise, SD 0.5. Drive excites itself.
- When D reaches threshold T a call starts. During the call M = D − T, and M sets call amplitude. "After 1.5 s, the time when D(t) reached the threshold, D(t) is reset to 0." "This reset represents the effect of self-vocalization on D(t)."
- When a partner call is heard: A′(t) = −bA(t) + I(t). A rises with input and decays; A subtracts from D′. "The auditory node, upon reception of a vocal signal from a conspecific, inhibits the activity in drive." "When that inhibition is weaker (e.g., following a lower amplitude, far away vocalization), the drive will reach threshold faster."
- Constants as extracted: α = 0.0003, T = 1000, b = 0.001, I = 0.0002 for a loud nearby call and 0.0001 for a soft distant one, input duration 3 s, Euler-Maruyama step 0.01, all initial values 0. The time unit of these constants was not stated in what could be read.
- Proposed anatomy: drive in "limbic structures such as the anterior cingulate cortex, amygdala"; motor in "periaqueductal gray area, reticular formation, nucleus retroambiguus". "Increasing arousal levels 'drive' the production of phee calls, whereas vocal feedback inhibits the drive."
- Behavioural test: responses to soft, "far-away" playback came at 3.11 ± 0.29 s, to loud "nearby" playback at 4.42 ± 0.67 s (P = 0.042, 8 sessions). Louder input, more inhibition, later reply.

### 3.3 Behavioural numbers in adult marmosets

- With periodic 75 dB noise bursts, 91.7 ± 6.3% of calls were started in the silent gaps (79% with 2 s gaps, 90% with 4 s, 97% with 8 s). Mean latency from noise offset to call onset: 1.54 s, 2.6 s and 4.78 s for 2, 4 and 8 s gaps. With unpredictable gaps the animals preferred long ones. Eight animals, 3880 calls (Roy 2011).
- About 90% of antiphonal calls come within 6 s of the preceding call (Miller 2009, citing Miller & Wang 2006). Answering stayed above the spontaneous rate when the simulated partner replied after 1, 3, 6 or 9 s, but not after 15 s (Miller 2009, 10 animals).
- Alone, the latency between calls peaks near 10 s; with a partner the distribution is bimodal, a fast response peak plus the 10 s peak (Liao 2018, 6 animals).
- Marmosets can cut off an ongoing call quickly after a sound, but only at periodic points within the call (Pomberger 2018, abstract).

### 3.4 The replication that failed, and the stochastic model (Grijseels, Fairbank & Miller 2024)

42 marmosets, 107 pair sessions, 11 614 calls (full text).

- Overlap avoidance replicated: only 234 of 11 450 calls (2.04%) were interrupted, less than in shuffled controls.
- No oscillation in cross- or auto-correlations. No positive phase-response correlation. Subsampling to the size of the 2013 study gave correlations from −0.74 to 0.81; 2.5% of subsamples reproduced the original result. The authors attribute the 2013 finding to small sample size.
- Most responses came about 3 s after the partner's call ended; the only factor predicting response delay was the responder's own calling rate.
- After its own call an animal has "a refractory period of about 2.2 s", then a rapid rise in call probability, back to baseline around 4.6 s after the offset. The authors read most of the refractory period as active suppression that leaves room for the partner.
- The probability of calling while the partner is calling falls as an exponential decay from call onset; most interruptions happen in the first 0.5 s and look like both animals starting at once.
- Their model multiplies four factors into a call probability per 0.2 s bin and draws calls at random: a slowly varying internal state, a decay of volubility over the session, the interruption curve, and the response curve. Removing the response curve was the only removal that changed inter-call intervals and conversation statistics.
- With three animals, one caller per session lowered its volubility, which needed an added social suppression term.

Plainly: "a partner's call suppresses calling while it lasts and raises it for a few seconds afterwards, and one's own call is followed by a refractory period" is supported by both camps. "The two animals are coupled oscillators that entrain" is supported by one small study and contradicted by one large one. Grade: suppression plus rebound (a) as behaviour in marmosets; coupled-oscillator entrainment (b), contested.

### 3.5 How marmoset infants develop turn-taking

- Ten infants recorded from postnatal day 1 for two months with a parent out of sight (Takahashi 2016, full text). The probability that an infant call overlaps a parental call falls from about 25% on day 1 to about 15% at two months. Corrected for falling call rate, infants avoid about 40% of chance overlaps in the first month and about 60% afterwards.
- **Self-monitoring.** Until about day 17 the interval between a parent's call and the infant's next call equals the interval between the infant's own spontaneous calls: the infant resets its timing on a parental call exactly as on its own. After that the reply interval becomes shorter than the spontaneous interval, as in adults. The authors' reading is that the infant at first "cannot distinguish its own call from another's".
- Parents' behaviour is constant across the period; infants' influence on parents (Granger causality) grows from nothing over the first month.
- The rate of contingent parental calls did not predict how fast an infant's overlap declined (Spearman ρ = −0.382, p = 0.279).
- Marmosets develop about 12 times faster than humans, so two marmoset months stand for about two human years (Takahashi 2016).
- Turn-taking is already present on day 1; hand-reared animals at 10 weeks overlapped about 33% of responses against about 25% in parent-reared ones and produced fewer antiphonal phee calls (Qi 2025, abstract and full text).
- In older animals, 4 to 12 months, young marmosets interrupted callers more than adults until eight months. Parents were less likely to answer after their own call had been interrupted: after an interruption the parent's call probability stayed at the spontaneous level for the whole 10 s window, whereas after uninterrupted calls it was raised for 4 to 5 s. Parents interrupted juveniles more when the juvenile used the wrong call type. The authors conclude the timing is learned and "not simply due to a general maturational process" (Chow 2015; 10 young animals, 53 363 calls).
- Contingent parental calls do speed up the *acoustic* maturation of infant calls (Takahashi 2015; Takahashi 2017, closed-loop playback to twins), without changing the overall rate of spontaneous calling (Takahashi 2017). Limiting parental contact leaves infant-type vocal behaviour in place into subadulthood (Gultekin & Hage 2017, 2018).

Grades: decline of overlap in infancy (a) in marmosets, three laboratories. Role of self-monitoring (b), one study. Parental contingency teaching the timing: disputed, see section 6.

### 3.6 The same mechanism in other animals

- **Insects and frogs** (a). Rhythms are kept by pacemakers and "a basic interactive algorithm, common to many species, yields the collective synchrony or alternation observed" (Greenfield 1994, abstract). The algorithm, as summarised by Ravignani, Verga & Greenfield 2019 (full text): "a caller inhibits his central rhythm generator when he hears a neighbor's call", then "rebounds from inhibition at the end of neighbor's call". Alternation or synchrony fall out of it depending on call rates (Greenfield, Tourtellot & Snedden 1997, abstract).
- **Meerkats** (b). Group-level overlap avoidance arises from "two individual-level rules: call inhibition over short timescales, which prevents mutual interference, and call excitation over longer timescales, which stimulates further group calling" (Demartsev 2018, abstract; the timescales in seconds were not found in the abstract).
- **Japanese macaques** (b). Median gap between callers 250 ms; overlap in 6 of 106 intervals; call intervals adjusted to the partner's response interval (Katsu 2018).
- **Singing mice** (b). Counter-singing on a sub-second time course; orofacial motor cortex paces it (Okobi 2019, abstract).

### 3.7 Equivalent evidence in humans

- An oscillator model of adult turn-taking exists as a proposal: readiness to speak cycles with the speaker's syllable rate and listeners are counter-phased (Wilson & Wilson 2005, abstract). Grade (c).
- Levinson & Torreira 2015 (full text) are sceptical: human synchronisation works "by means of temporal estimation" rather than by resetting, and "given the highly variable lengths of turns, nothing like the firefly mechanism can work in conversation", while leaving "room for a low level metronome".
- Adult gaps of about 200 ms are shorter than the 600 ms or more needed to plan a word, so adults plan while listening and predict the end of the turn (Levinson & Torreira 2015; Bögels 2015; Castellucci 2022). This is a different, predictive mechanism layered on top and not available to a young infant.
- In infants, the pattern "vocalise, pause as if to listen, repeat if unanswered" (Masataka 1993) and the mutual adjustment of switching pauses at 4 and 9 months (Beebe 1988; Jasnow & Feldstein 1986; Jaffe 2001) are the closest behavioural equivalents. No infant study was found that fits a drive-inhibition or phase-response model to infant data.

---

## 4. Neural substrates

### 4.1 Brainstem pattern generation and the periaqueductal grey gate: (a)

- Lesions of the periaqueductal grey (PAG) and adjacent tegmentum abolish species-specific calls in squirrel monkeys. The PAG receives from anterior cingulate, amygdala, hypothalamus and other call-eliciting areas and projects to the laryngeal motor nucleus. It is "a necessary relay station" on the output side (Jürgens & Pratt 1979, abstract).
- The motor-coordinating network in pons and medulla "needs a facilitatory input from the periaqueductal grey" to vocalise (Jürgens 2002, abstract). A pathway from anterior cingulate through PAG to the reticular formation "controls the readiness to vocalize"; the PAG "is involved in vocal gating" (Jürgens 2009, abstract).
- A vocal pattern generator for frequency-modulated calls sits in the ventrolateral pontine brainstem (Hage & Jürgens 2006a, abstract).
- In mice a dedicated PAG population is necessary and sufficient for ultrasonic calls and gates downstream patterning circuits (Tschida 2019). Two inhibitory forebrain inputs act on it in opposite directions: preoptic neurons inhibit local PAG interneurons and so release calling; amygdala neurons inhibit the PAG call neurons directly and transiently suppress calling (Michael 2020). Call loudness and bout length are set by how strongly and for how long the local inhibitory clamp is lifted (Chen 2021).
- In humans PAG lesions cause mutism; the PAG drives the nucleus retroambiguus, which controls the muscles that set subglottal pressure (Holstege & Subramanian 2016; Holstege 2014).

The gate is tonically inhibited and opened by disinhibition, and a separate input can clamp it shut. Whether the sound of another voice reaches this gate through the amygdala path has **not been tested** in the papers read; that link is an inference, grade (c).

### 4.2 Anterior cingulate and medial frontal initiation: (a) for involvement, (b) for the dynamics

- Spontaneous isolation calls in squirrel monkeys depend on a band of rostral limbic cortex including area 24 (MacLean & Newman 1988, abstract). Voluntary initiation and suppression are attributed to mediofrontal cortex including anterior cingulate (Jürgens 2002).
- In macaques trained to call on cue, anterior cingulate neurons ramp up and the call starts when the ramp reaches a threshold; ventrolateral prefrontal onset activity tracks reaction time (Gavrilov, Hage & Nieder 2017, abstract; Hage & Nieder 2013). This is the single-neuron counterpart of a drive that rises to threshold.
- In marmosets exchanging calls, frontal and premotor neurons change firing before and during self-initiated calls (Miller 2015; Roy, Zhao & Wang 2016; Zhao & Wang 2023). The state of frontal cortex *before* a partner's call is heard predicts whether the animal will answer, and the change scales with conversation length (Nummela 2017, abstract).
- The dual-network proposal: a subcortical primary vocal network under volitional control from prefrontal cortex, the latter immature in infants (Hage & Nieder 2016, abstract). Grade (c) as a developmental account.

### 4.3 Suppression of auditory cortex by one's own vocalising: (a)

- Marmoset auditory cortex: most neurons are suppressed during the animal's own calls, and the suppression "began several hundred milliseconds prior to the onset of vocalization" (Eliades & Wang 2003, abstract). Suppressed neurons are *more* sensitive to altered feedback (Eliades & Wang 2008), and auditory cortex activity drives feedback corrections of the voice (Eliades & Tsunada 2018).
- Two components: a brief phasic suppression locked to each phrase that is untuned (gating), and a sustained tonic suppression tuned to vocal frequencies (prediction) (Tsunada, Wang & Eliades 2024, abstract).
- Circuit in mice: secondary motor cortex neurons drive local inhibitory interneurons in auditory cortex before and during movement (Schneider, Nelson & Mooney 2014, abstract).
- Humans: responses to one's own speech are weaker than to playback of the same speech (Houde 2002, MEG, abstract); suppression varies across auditory cortex sites (Flinker 2010).
- Brainstem: neurons around the superior olive change firing before vocal onset and respond to external sound; candidates for the Lombard reflex (Hage, Jürgens & Ehret 2006b; Luo, Hage & Moss 2018).
- Songbirds: auditory responses of the premotor nucleus HVC are gated by behavioural state (Schmidt & Konishi 1998), and distorted feedback does not reach HVC premotor synapses during singing (Hamaguchi 2014, abstract).

### 4.4 The other direction, sound stopping or delaying vocal output: (a) in songbirds, (a) behaviourally in marmosets, not found in human infants

- Zebra finches exchange short calls with stereotyped latencies; 50% of answers fall within a window of 249 ± 23 ms, medians across birds 249 to 466 ms. When a robot partner places calls where they would collide, birds shift their timing within minutes. Cutting the output of the forebrain song system destroys both the precision and the avoidance (Benichov 2016, full text).
- Inside HVC, inhibitory interneurons fire before the premotor neurons that drive a call. Blocking inhibition there with gabazine shortened reply latency from 266 ± 51 ms (249 ± 68 ms with saline) to 166 ± 25 ms and impaired the bird's ability to avoid overlapping its partner; call rate, pitch and duration did not change (Benichov & Vallentin 2020, full text, 5 birds).
- A feedforward-inhibition model constrained by recordings: the partner's call drives HVC interneurons within about 35 ms of call onset; premotor bursts due 25 to 55 ms after onset are suppressed, later ones delayed. The same circuit gives timed replies and suppression while listening (Norton 2022, full text).
- Duetting wrens alternate syllables at 2 to 5 Hz. In paired recordings HVC activity rises for the bird's own syllables and falls while it hears the partner's (median change about −1.9 spikes/s in females, −1.1 in males). Under urethane, which blocks GABA transmission, the same sounds excite HVC. The authors propose inhibition followed by rebound (Coleman 2021; review Coleman, Day & Fortune 2022). In sparrow-weavers the partner's first syllable resets the rhythm of the initiating bird's premotor discharge (Hoffmann 2019, abstract).
- Nightingales start songs in the silent intervals of playback (Brumm 2006).
- In zebra finches the inhibition that a heard song evokes in premotor neurons matures with *learning*, not with age (Vallentin 2016, abstract).
- Marmosets: section 3.3.
- Human infants: no study of a neural pathway by which heard speech delays vocal output was found.

### 4.5 Arousal systems: (b)

Section 1.3. Arousal measured by heart rate predicts call probability and latency in marmosets (Borjon 2016; Liao 2018), couples vocal and locomotor development (Gustison 2019), and clusters vocalisations in human infants (Wass 2022). In juvenile zebra finches a singing tutor excites dopamine neurons in the PAG that project to HVC (Tanaka 2018), and social tutoring activates noradrenergic and dopaminergic midbrain neurons (Chen 2016). No paper read here identifies the arousal nucleus that sets the call drive in primates.

---

## 5. Contingent social feedback

### 5.1 What caregivers do

- Caregivers respond to 30 to 50% of non-cry vocalisations (Goldstein 2009, abstract); in free play to over 70% (Gros-Louis 2006, abstract).
- Most maternal responses at 2 to 4 months come about 1 s after the infant's signal (Gratier 2015, introduction). In a study of 150 dyads at 4 months a 3 s window captured contingencies best (Van Egeren 2001, abstract).
- Even in the first three months caregivers take turns with protophones and talk over cries (Yoo 2018, abstract).
- Responses are selective: speech-like sounds get more responses, and a child's next sound is more likely to be speech-like if the last one was answered (Warlaumont 2014, 13 836 hours of recordings). Contingent parental speech is simpler than non-contingent speech (Elmlinger 2019).

### 5.2 What contingency changes

- **Quality: (a).** Contingent, not yoked non-contingent, maternal responses push 8-month-olds' babbling toward more mature forms within minutes (Goldstein 2003), and 9.5-month-olds take over the phonological pattern of the contingent speech (Goldstein & Schwade 2008). At 3 months, turn-taking responses raise the proportion of syllabic sounds (Bloom, Russell & Wassenberg 1987), but only when the adult's response contains speech (Bloom 1988). Marmosets: Takahashi 2015, 2017. Zebra finches: contingent non-vocal feedback from females improves song learning with singing rate unchanged (Carouso-Peck & Goldstein 2019).
- **Timing: (b).** With contingent turn-taking 3-month-olds adopt "a speak-listen pattern" (Bloom 1988). With random responding infants vocalise in bursts; the inter-burst interval tracks the mother's recent response latency (Masataka 1993). Infant inter-vocalisation intervals shorten after a response (Ritwika 2020).
- **Quantity: disputed.** Early conditioning studies reported that contingent adult responses raise vocal rate in 3-month-olds, with a decline toward baseline in extinction, but the result was "equivocal since the reinforcing stimuli, per se, may have acted as social releasers" (Weisberg 1963, abstract, describing Rheingold 1959, which is listing only). Bloom & Esposito 1975 on the proper controls is listing only. Masataka 1993 found that rate "was not significantly influenced by the contingency". Marmoset and finch closed-loop studies found no effect on rate (Takahashi 2017; Carouso-Peck & Goldstein 2019). Iyer 2016 found no difference in volubility between a talking and a silent parent. A 2026 study with an artificial agent found a linear rise in vocalising while contingent vocal imitation was on (Diprossimo 2026, abstract).

### 5.3 Do infants learn the contingency?

- Still-face: when an adult who has been interacting goes still, infants vocalise *more*. 5-month-olds show an extinction burst, above baseline in the first 15 s and from 30 to 60 s, baseline being 0.79 vocalisations per 15 s; the size of the burst predicts comprehension at 13 months (Goldstein, Schwade & Bornstein 2009, 38 infants). Replicated at about 6 months with volubility staying high in the reunion (Franklin 2014, 24 dyads).
- 2-month-olds do not show the vocal burst, 5-month-olds do, and its size follows the caregiver's own responsiveness. "Infants thus learn the social efficacy of their vocalizations between 2 and 5 months of age" (Elmlinger 2023, abstract).
- Non-distress vocal "bids" toward a still mother appear at 2 months and are best predicted by the mother's contingent vocal responsiveness (Bigelow & Power 2016, 2022, abstracts).
- Infants detect a 1 s delay in the mother's audio-visual response at 3 and 6 months (Striano 2006) and a 3 s delay (Henning & Striano 2011); 2-month-olds react to a replay of the mother in place of the live mother (Nadel 1999); 1-month-olds do not discriminate contingent from non-contingent interaction, 3-month-olds do (Striano 2005).
- Classic still-face references that are listing only: Tronick 1978; Mesman 2009 meta-analysis; Delgado 2002.

Grade: (a) that infants expect a reply by 5 months and that the expectation depends on experience.

### 5.4 What is the reward signal?

Not established in human infants.

- Behaviourally the effective event is a response that is *contingent in time* on the infant's sound, vocal or not (Goldstein 2003 used non-imitative social responses; Carouso-Peck 2019 used visual responses in finches). For phonological learning the response must contain speech (Bloom 1988; Goldstein & Schwade 2008).
- Proposal, (c): social motivation circuitry tied to cortico-striatal learning (Syal & Finlay 2011). A spiking model of babbling uses a dopamine-like reward gating spike-timing plasticity, with reward given for more salient sounds (Warlaumont & Finnegan 2016).
- Songbirds, (b): dopamine neurons report better or worse than expected song outcome (Gadagkar 2016); a tutor's presence releases dopamine in HVC (Tanaka 2018).

---

## 6. What changes with age so that an infant waits and responds

Evidence that it is **learned from contingency**:

- The expectation of a reply appears between 2 and 5 months and scales with caregiver responsiveness (Elmlinger 2023; Bigelow & Power 2016).
- Infants adjust their pause after their own vocalisation to the mother's recent response latency (Masataka 1993), and infant latency tracks adult latency across studies (Nguyen 2022).
- Marmoset parents withhold replies after being interrupted (Chow 2015). Hand-reared marmosets overlap more (Qi 2025). Zebra finches learn within minutes to move their call away from a colliding one (Benichov 2016).
- In zebra finches premotor inhibition by heard song matures with learning and not with age (Vallentin 2016).

Evidence that it is **maturation or an early-present capacity**:

- Offset-contingent vocalising and latched turns are present in newborns and at 2 to 3 months (Dominguez 2016; Gratier 2015); marmosets turn-take on day 1 (Qi 2025).
- In marmoset infants the decline in overlap follows the emergence of self-monitoring at about day 17 and is unrelated to how often parents reply contingently (Takahashi 2016).
- Vocal coordination measures show no developmental change from 3 to 12 months in one longitudinal study (Northrup & Iverson 2020), and the meta-analysis finds no clear change in latency with age (Nguyen 2022).
- Body growth alone shifts marmoset call types and durations (Zhang & Ghazanfar 2018).
- The proposal that prefrontal control over the subcortical vocal system matures late (Hage & Nieder 2016).

Where the literature disagrees:

1. **Parental feedback and the timing of turn-taking in marmosets.** Chow, Mitchell & Miller 2015: learned, parents give feedback on errors. Takahashi, Fenley & Ghazanfar 2016: "no evidence that the rate of contingent calls by parents influences the development of vocal turn-taking", and they point out the 2015 study looked at juveniles, not infants. The two studied different ages (4 to 12 months against 0 to 2 months) and different measures (response after an interruption against overall rate of contingent calls), so both can hold.
2. **Coupled oscillators or stochastic calling.** Takahashi 2013 against Grijseels 2024, section 3.4.
3. **When human infants shift from overlap to alternation.** By 12 to 18 weeks (Ginsburg & Kilbourne 1988, 3 dyads); no change between 2 to 3 and 4 to 5 months (Gratier 2015); only after 5 months (Hilbrink 2015); between 4 and 7 months (Harder 2015); gradually to 18 months (Elias & Broerse 1996). Rutter & Durkin 1987 (listing only) is reported by Gratier 2015 and Hilbrink 2015 as finding more interruption before 24 months.
4. **Who makes the turn-taking.** Mothers fit their speech around a randomly vocalising infant (Elias, Hayes & Broerse 1986; Snow 1977 holds that mothers impose a conversational model), against infants timing their own turns (Hilbrink 2015 random-distribution test; Gratier 2015 latching; Bornstein 2015).
5. **Does infant timing improve at all in the first year?** Hilbrink 2015: infants get slower at 9 months. Nguyen 2022 and Northrup & Iverson 2020: no clear trend.
6. **Does contingency raise the amount of vocalising?** Section 5.2.

Most defensible reading: both. The capacity to be inhibited by a voice and to start at its offset is there from the start and is cheap (it exists in frogs). What is learned is what counts as "another's voice" as opposed to one's own, the expectation of a reply after one's own sound, and how long to wait. Human evidence for the "maturing inhibition" half is indirect; no infant study measuring inhibitory control against turn-taking was found.

---

## Candidate mechanisms for a simulated learner

Ordered from best to least established. For each: the biology, the smallest faithful form, what should emerge, and what would be a hand-wired shortcut.

### 1. A gated vocal drive with discharge and refractory period

**Biology.** Vocalising needs a facilitating input to the brainstem pattern generator through the PAG gate (Jürgens & Pratt 1979; Jürgens 2009; Tschida 2019), opened by disinhibition and closable by a separate inhibitory input (Michael 2020; Chen 2021). The readiness to call comes from limbic and medial frontal drive; anterior cingulate activity ramps to a threshold before a call (Gavrilov 2017). Call probability follows arousal, which climbs before a call and falls after it (Borjon 2016; Wass 2022). Anatomy (a); ramp-to-threshold and arousal dynamics (b).

**Smallest form.**
- State: drive D, a slow arousal variable R, a gate that is closed by default.
- D integrates R plus noise and leaks. When D crosses threshold the gate opens for one utterance; the utterance resets D toward zero and lowers R (Choi 2015 reset; Borjon 2016 deceleration).
- R fluctuates slowly and is raised by salient events. The learner's own brain state should supply R; it should not be a clock.
- Published time constants: marmoset inter-call interval about 10 s alone, arousal rise about 23 s before a call and fall about 8 s after, slow autonomic rhythm 0.1 Hz (Borjon 2016); refractory period about 2.2 s after own call (Grijseels 2024). Human infant: about 4 utterances per minute alone (Oller 2019), 5 to 7% of time vocalising (Iyer 2016), utterance about 0.9 s, pause to the next own utterance 0.3 to 1.2 s inside a bout (Gratier 2015; Hilbrink 2015), jaw cycle about 3 per second inside an utterance (Dolata 2008).

**Should emerge.** Bouts of a few utterances separated by long silences; a rate of a few per minute that rises and falls with arousal; silence when arousal is low; no two utterances closer than the refractory period.

**Wrong shortcut.** A timer that emits a sound every N milliseconds. A fixed probability per tick. A hard-coded rate of 4 per minute would be the same mistake with a better number.

### 2. Inhibition of the drive (or the premotor stage) by a heard voice, then release

**Biology.** Heard calls excite inhibitory interneurons in the premotor nucleus; removing that inhibition makes replies faster and causes overlap (Benichov & Vallentin 2020; Norton 2022). Premotor activity drops while the partner sings (Coleman 2021). Marmosets place 92% of calls in silent gaps (Roy 2011) and reply later to louder calls (Choi 2015). Frogs and insects inhibit their rhythm generator during a neighbour's call and rebound at its end (Greenfield 1994; Ravignani 2019). Meerkats: short-timescale inhibition plus longer-timescale excitation (Demartsev 2018). Grade (a) in songbirds, insects and frogs; (a) behaviourally in marmosets; in human infants only the outcome is known.

**Smallest form.** Choi 2015: D′ = α|D| − A + noise; A′ = −bA + I, with I the heard voice. A is a leaky trace of voice-like input and subtracts from the growth of D. Add what the infant and meerkat data require: the same input also raises arousal R on a slower timescale, so a voice both holds the drive down now and raises it for the seconds after.
- Time constants: zebra finch inhibition reaches the premotor stage about 35 ms after call onset, suppression window 25 to 55 ms, reply latency about 250 ms (Norton 2022; Benichov 2016, 2020). Marmoset replies about 3 s after offset, raised call probability until about 4.6 s (Grijseels 2024), 3.1 s to soft and 4.4 s to loud calls (Choi 2015), exchange broken by a 15 s delay (Miller 2009). Human infants: reply gaps with medians of 0.3 to 0.9 s at 3 months (Hilbrink 2015), about 1 s on average across studies (Nguyen 2022), contingency window 2 s (Bornstein 2015).

**Should emerge.** Fewer onsets while a voice is heard; a burst of onsets shortly after it ends; later replies after louder or longer input; more vocalising overall in the presence of a responsive voice than in silence; some overlap, especially when inhibition is weak. A third of infant vocalisations overlap in the first months, so an early learner that overlaps is faithful and one that never overlaps is not.

**Wrong shortcut.** "If sound is present, do not vocalise." A voice-activity detector that mutes the mouth. A fixed reply delay after detected offset. Zero overlap by construction.

### 3. Corollary discharge: telling one's own voice from another's

**Biology.** A motor-related signal suppresses auditory cortex from several hundred milliseconds before one's own vocalisation, with an untuned phasic gate and a tuned tonic prediction (Eliades & Wang 2003; Tsunada 2024; Schneider 2014; Houde 2002). Infant marmosets at first treat a parent's call like their own, resetting their timing on it, and begin to answer faster than their spontaneous interval only after about day 17 (Takahashi 2016). Existence (a); role in the development of turn-taking (b), one study.

**Smallest form.** The motor command sends a copy to the ear that subtracts the predicted sound of the own voice, starting before the sound and lasting through it. Only the residual drives the inhibitory trace A of mechanism 2 and the "other voice" input to arousal. The prediction is learned from hearing oneself, so early on it is poor.
- Time constants: suppression begins several hundred milliseconds before onset (Eliades & Wang 2003); premotor activity leads the syllable by about 50 ms in wrens (Coleman 2021).

**Should emerge.** Early: the learner's own voice inhibits its own drive like anyone else's, and a partner's voice resets it like its own, so the reply interval equals the spontaneous interval. Later, as the prediction improves: own voice no longer counts as a partner, replies become faster than the spontaneous interval, overlap falls. With the speaker feeding the microphone, this is also what stops the learner from answering itself.

**Wrong shortcut.** Muting the microphone while the mouth is active. A flag "this sound is mine" passed from mouth to ear. Either one removes the residual that the brain uses to monitor its own voice (Eliades & Wang 2008).

### 4. Contingent replies as reinforcement of waiting

**Biology.** Caregivers answer about 1 s after a sound (Gratier 2015), selectively (Warlaumont 2014). Infants come to expect it between 2 and 5 months and vocalise more when the reply fails (Goldstein 2009; Elmlinger 2023). Infants pause after vocalising "as if to listen" and match the pause to the mother's recent latency (Masataka 1993). Marmoset parents do not answer calls that interrupted them (Chow 2015). Learned expectation (a); effect on timing (b); effect on turn-taking maturation rate disputed (Takahashi 2016); dopamine as the signal (c) in infants, (b) in songbirds.

**Smallest form.**
- An eligibility trace of the learner's own recent utterance, lasting a few seconds. Published windows: 1 s (Gratier 2015), 2 s (Bornstein 2015), 3 s (Van Egeren 2001), 5 s (Takahashi 2016).
- A reinforcement signal when another voice starts inside that window. It strengthens whatever state preceded the utterance, including "the other voice had stopped" and "my drive was held back".
- A prediction of the reply. When the predicted reply does not come, arousal rises (the extinction burst).
- Plastic quantities: the strength of voice-evoked inhibition (mechanism 2) and the length of the post-utterance pause. In the finch that inhibition matures with learning, not age (Vallentin 2016).

**Should emerge.** Utterances that begin in silence get answered more often, so starting in silence becomes more frequent. The pause after an own utterance drifts toward the partner's usual latency. When a responsive partner goes silent the learner vocalises more for about a minute and then less. With a partner who answers at random the learner falls back into bursts.

**Wrong shortcut.** Rewarding silence during speech directly. A scripted "wait N seconds for a reply". Scoring turn-taking and optimising the score. The reward has to be the partner's reply, nothing the designer computes about timing.

### 5. Phase-resetting oscillator and entrainment between partners

**Biology.** Established in rhythmic insects and frogs (Greenfield 1994; Greenfield 1997). Claimed for marmosets (Takahashi 2013) and not replicated in a larger sample (Grijseels 2024). Proposed for human conversation (Wilson & Wilson 2005) and doubted (Levinson & Torreira 2015). Grade (b), contested, in primates; (c) in humans.

**Smallest form.** Mechanisms 1 and 2 together already are a relaxation oscillator with inhibitory resetting: drive rises, fires, resets, and a heard call pushes the phase back. Nothing further needs to be added. Whether the pair then entrains depends on how regular the drive is. With a noisy, arousal-driven drive the result is the stochastic pattern of Grijseels 2024; with a steady drive it is the periodic pattern of Takahashi 2013.

**Should emerge, if at all.** A weak dependence of the learner's next interval on how fast the partner replied.

**Wrong shortcut.** Adding a clock with a period and a phase-reset rule on top. Tuning the system to show a 10 s periodicity or a positive phase-response correlation as a target.

### A check that needs no new mechanism

For any of the above, the measurements used on infants and marmosets can be run on the learner as observations: utterances per minute alone; percent of time vocalising; share of utterance onsets that fall inside a partner's voice against a shuffled control (Takahashi 2016; Hilbrink 2015); distribution of gaps from partner offset to learner onset; spontaneous interval against reply interval (Takahashi 2016); the still-face minute (Goldstein 2009).

---

## References

Status: **full text**, **abstract**, or **listing only**. "PMC page" means read through the page-fetch tool.

### Infant spontaneous vocalisation and its drivers

- Abney DH, Warlaumont AS, Oller DK, Wallot S, Kello CT (2017). Multiple coordination patterns in infant and adult vocalizations. Infancy 22(4):514-539. doi:10.1111/infa.12165. Full text.
- Boliek CA, Hixon TJ, Watson PJ, Morgan WJ (1996). Vocalization and breathing during the first year of life. J Voice 10(1):1-22. doi:10.1016/s0892-1997(96)80015-4. Abstract.
- Borjon JI, Sahoo MK, Rhodes KD, Lipschutz R, Bick JR (2024). Recognizability and timing of infant vocalizations relate to fluctuations in heart rate. Proc Natl Acad Sci USA 121(52). doi:10.1073/pnas.2419650121. Abstract.
- Borjon JI, Takahashi DY, Cervantes DC, Ghazanfar AA (2016). Arousal dynamics drive vocal production in marmoset monkeys. J Neurophysiol 116(2):753-764. doi:10.1152/jn.00136.2016. Abstract and PMC page.
- Caskey M, Stephens B, Tucker R, Vohr B (2011). Importance of parent talk on the development of preterm infant vocalizations. Pediatrics 128(5):910-916. doi:10.1542/peds.2011-0609. Abstract.
- Dolata JK, Davis BL, MacNeilage PF (2008). Characteristics of the rhythmic organization of vocal babbling: implications for an amodal linguistic rhythm. Infant Behav Dev 31(3):422-431. doi:10.1016/j.infbeh.2007.12.014. Abstract.
- Franklin B, Warlaumont AS, Messinger D, Bene E, Iyer SN, Lee CC, Lambert B, Oller DK (2014). Effects of parental interaction on infant vocalization rate, variability and vocal type. Lang Learn Dev 10(3):279 ff. (PubMed prints the end page as 2996). doi:10.1080/15475441.2013.849176. Full text.
- Gustison ML, Borjon JI, Takahashi DY, Ghazanfar AA (2019). Vocal and locomotor coordination develops in association with the autonomic nervous system. eLife 8:e41853. doi:10.7554/eLife.41853. Abstract.
- Iyer SN, Denson H, Lazar N, Oller DK (2016). Volubility of the human infant: effects of parental interaction (or lack of it). Clin Linguist Phon 30(6):470-488. doi:10.3109/02699206.2016.1147082. Full text.
- Jhang Y, Oller DK (2017). Emergence of functional flexibility in infant vocalizations of the first 3 months. Front Psychol 8:300. doi:10.3389/fpsyg.2017.00300. Abstract.
- Jones SJ, Moss HA (1971). Age, state, and maternal behavior associated with infant vocalizations. Child Dev 42(4):1039-1051. PMID 5157100. Listing only.
- Liao DA, Zhang YS, Cai LX, Ghazanfar AA (2018). Internal states and extrinsic factors both determine monkey vocal production. Proc Natl Acad Sci USA 115(15):3978-3983. doi:10.1073/pnas.1722426115. Abstract and PMC page.
- Long HL, Bowman DD, Yoo H, Burkhardt-Reed MM, Bene ER, Oller DK (2020). Social and endogenous infant vocalizations. PLoS One 15(8):e0224956. doi:10.1371/journal.pone.0224956. Full text.
- Lynch MP, Oller DK, Steffens ML, Buder EH (1995). Phrasing in prelinguistic vocalizations. Dev Psychobiol 28(1):3-25. doi:10.1002/dev.420280103. Abstract.
- MacNeilage PF (1998). The frame/content theory of evolution of speech production. Behav Brain Sci 21(4):499-511. doi:10.1017/s0140525x98001265. Abstract.
- Oller DK, Buder EH, Ramsdell HL, Warlaumont AS, Chorna L, Bakeman R (2013). Functional flexibility of infant vocalization and the emergence of language. Proc Natl Acad Sci USA 110(16):6318-6323. doi:10.1073/pnas.1300337110. Abstract.
- Oller DK, Griebel U, Iyer SN, Jhang Y, Warlaumont AS, Dale R, Call J (2019). Language origins viewed in spontaneous and interactive vocal rates of human and bonobo infants. Front Psychol 10:729. doi:10.3389/fpsyg.2019.00729. Full text.
- Oller DK, Ramsay G, Bene E, Long HL, Griebel U (2021). Protophones, the precursors to speech, dominate the human infant vocal landscape. Philos Trans R Soc B 376(1836):20200255. doi:10.1098/rstb.2020.0255. Full text.
- Pretzer GM, Lopez LD, Walle EA, Warlaumont AS (2019). Infant-adult vocal interaction dynamics depend on infant vocal type, child-directedness of adult speech, and timeframe. Infant Behav Dev 57:101325. doi:10.1016/j.infbeh.2019.04.007. Abstract.
- Ritwika VPS, Pretzer GM, Mendoza S, Shedd C, Kello CT, Gopinathan A, Warlaumont AS (2020). Exploratory dynamics of vocal foraging during infant-caregiver communication. Sci Rep 10:10469. doi:10.1038/s41598-020-66778-0. Full text.
- Shimada Y (2012). Infant vocalization when alone: possibility of early sound playing. Int J Behav Dev 36(6):407-412. doi:10.1177/0165025411431408. Abstract.
- Wass S, Phillips E, Smith C, Fatimehin EOOB, Goupil L (2022). Vocal communication is tied to interpersonal arousal coupling in caregiver-infant dyads. eLife 11:e77399. doi:10.7554/eLife.77399. Full text.
- Zhang YS, Ghazanfar AA (2016). Perinatally influenced autonomic system fluctuations drive infant vocal sequences. Curr Biol 26(10):1249-1260. doi:10.1016/j.cub.2016.03.023. Abstract.
- Zhang YS, Ghazanfar AA (2018). Vocal development through morphological computation. PLoS Biol 16(2):e2003933. doi:10.1371/journal.pbio.2003933. Abstract.

### Human turn-taking and its development

- Anderson BJ (1977; usually cited as Anderson, Vietze & Dokecki; PubMed lists the first author only). Reciprocity in vocal interactions of mothers and infants. Child Dev 48(4):1676-1681. PMID 608379. Abstract.
- Bateson MC (1975). Mother-infant exchanges: the epigenesis of conversational interaction. Ann N Y Acad Sci 263:101-113. doi:10.1111/j.1749-6632.1975.tb41575.x. Listing only.
- Beebe B, Alson D, Jaffe J, Feldstein S, Crown C (1988). Vocal congruence in mother-infant play. J Psycholinguist Res 17(3):245-259. doi:10.1007/BF01686358. Abstract.
- Bloom K (1988). Quality of adult vocalizations affects the quality of infant vocalizations. J Child Lang 15(3):469-480. doi:10.1017/s0305000900012502. Abstract.
- Bloom K, Russell A, Wassenberg K (1987). Turn taking affects the quality of infant vocalizations. J Child Lang 14(2):211-227. doi:10.1017/s0305000900012897. Abstract.
- Bögels S, Magyari L, Levinson SC (2015). Neural signatures of response planning occur midway through an incoming question in conversation. Sci Rep 5:12881. doi:10.1038/srep12881. Abstract.
- Bornstein MH, Putnick DL, Cote LR, Haynes OM, Suwalsky JT (2015). Mother-infant contingent vocalizations in 11 countries. Psychol Sci 26(8):1272-1284. doi:10.1177/0956797615586796. Full text.
- Casillas M, Bobb SC, Clark EV (2016). Turn-taking, timing, and planning in early language acquisition. J Child Lang 43(6):1310-1337. doi:10.1017/S0305000915000689. Abstract.
- Castellucci GA, Kovach CK, Howard MA, Greenlee JDW, Long MA (2022). A speech planning network for interactive language use. Nature 602(7895):117-122. doi:10.1038/s41586-021-04270-z. Abstract.
- Dominguez S, Devouche E, Apter G, Gratier M (2016). The roots of turn-taking in the neonatal period. Infant Child Dev 25(3):240-255. doi:10.1002/icd.1976. Abstract.
- Elias G, Broerse J (1996). Developmental changes in the incidence and likelihood of simultaneous talk during the first two years: a question of function. J Child Lang 23(1):201-217. doi:10.1017/s0305000900010151. Abstract.
- Elias G, Hayes A, Broerse J (1986). Maternal control of co-vocalization and inter-speaker silences in mother-infant vocal engagements. J Child Psychol Psychiatry 27(3):409-415. doi:10.1111/j.1469-7610.1986.tb01842.x. Abstract.
- Ginsburg GP, Kilbourne BK (1988). Emergence of vocal alternation in mother-infant interchanges. J Child Lang 15(2):221-235. doi:10.1017/s0305000900012344. Abstract.
- Gratier M, Devouche E, Guellai B, Infanti R, Yilmaz E, Parlato-Oliveira E (2015). Early development of turn-taking in vocal interaction between mothers and infants. Front Psychol 6:1167. doi:10.3389/fpsyg.2015.01167. Full text.
- Harder S, Lange T, Hansen GF, Væver M, Køppe S (2015). A longitudinal study of coordination in mother-infant vocal interaction from age 4 to 10 months. Dev Psychol 51(12):1778-1790. doi:10.1037/a0039834. Abstract.
- Hilbrink EE, Gattis M, Levinson SC (2015). Early developmental changes in the timing of turn-taking: a longitudinal study of mother-infant interaction. Front Psychol 6:1492. doi:10.3389/fpsyg.2015.01492. Full text.
- Jaffe J, Beebe B, Feldstein S, Crown CL, Jasnow MD (2001). Rhythms of dialogue in infancy: coordinated timing in development. Monogr Soc Res Child Dev 66(2):1-132. PMID 11428150. Abstract.
- Jasnow M, Feldstein S (1986). Adult-like temporal characteristics of mother-infant vocal interactions. Child Dev 57(3):754-761. PMID 3720401. Abstract.
- Levinson SC (2016). Turn-taking in human communication: origins and implications for language processing. Trends Cogn Sci 20(1):6-14. doi:10.1016/j.tics.2015.10.010. Listing only.
- Levinson SC, Torreira F (2015). Timing in turn-taking and its implications for processing models of language. Front Psychol 6:731. doi:10.3389/fpsyg.2015.00731. Full text.
- Masataka N (1993). Effects of contingent and noncontingent maternal stimulation on the vocal behaviour of three- to four-month-old Japanese infants. J Child Lang 20(2):303-312. doi:10.1017/s0305000900008291. Abstract.
- Nguyen V, Versyp O, Cox C, Fusaroli R (2022). A systematic review and Bayesian meta-analysis of the development of turn taking in adult-child vocal interactions. Child Dev 93(4):1181-1200. doi:10.1111/cdev.13754. Full text.
- Northrup JB, Iverson JM (2020). The development of mother-infant coordination across the first year of life. Dev Psychol 56(2):221-236. doi:10.1037/dev0000867. Full text.
- Papoušek M, Papoušek H (1989). Forms and functions of vocal matching in interactions between mothers and their precanonical infants. First Language 9(6):137-157. doi:10.1177/014272378900900603. Abstract.
- Rosenthal M (1982). Vocal dialogues in the neonatal period. Dev Psychol 18(1):17-21. doi:10.1037/0012-1649.18.1.17. Listing only.
- Rutter DR, Durkin K (1987). Turn-taking in mother-infant interaction: an examination of vocalizations and gaze. Dev Psychol 23(1):54-61. doi:10.1037/0012-1649.23.1.54. Listing only.
- Snow CE (1977). The development of conversation between mothers and babies. J Child Lang 4(1):1-22. doi:10.1017/s0305000900000453. Abstract.
- Stern DN, Jaffe J, Beebe B, Bennett SL (1975). Vocalizing in unison and in alternation: two modes of communication within the mother-infant dyad. Ann N Y Acad Sci 263:89-100. doi:10.1111/j.1749-6632.1975.tb41574.x. Listing only.
- Stivers T, Enfield NJ, Brown P, Englert C, Hayashi M, Heinemann T, Hoymann G, Rossano F, de Ruiter JP, Yoon KE, Levinson SC (2009). Universals and cultural variation in turn-taking in conversation. Proc Natl Acad Sci USA 106(26):10587-10592. doi:10.1073/pnas.0903616106. Abstract.
- Van Egeren LA, Barratt MS, Roach MA (2001). Mother-infant responsiveness: timing, mutual regulation, and interactional context. Dev Psychol 37(5):684-697. doi:10.1037/0012-1649.37.5.684. Abstract.
- Webster RL (1969). Selective suppression of infants' vocal responses by classes of phonemic stimulation. Dev Psychol 1(4):410-414. doi:10.1037/h0027690. Listing only.
- Wilson M, Wilson TP (2005). An oscillator model of the timing of turn-taking. Psychon Bull Rev 12(6):957-968. doi:10.3758/bf03206432. Abstract.
- Yoo H, Bowman DA, Oller DK (2018). The origin of protoconversation: an examination of caregiver responses to cry and speech-like vocalizations. Front Psychol 9:1510. doi:10.3389/fpsyg.2018.01510. Abstract.

### Marmosets and other mammals: turn-taking, models, development

- Choi JY, Takahashi DY, Ghazanfar AA (2015). Cooperative vocal control in marmoset monkeys via vocal feedback. J Neurophysiol 114(1):274-283. doi:10.1152/jn.00228.2015. Abstract and PMC page.
- Chow C, Mitchell J, Miller C (2015). Vocal turn-taking in a non-human primate is learned during ontogeny. Proc R Soc B 282(1807):20150069. doi:10.1098/rspb.2015.0069. Abstract and PMC page.
- Demartsev V, Strandburg-Peshkin A, Ruffner M, Manser M (2018). Vocal turn-taking in meerkat group calling sessions. Curr Biol 28(22):3661-3666. doi:10.1016/j.cub.2018.09.065. Abstract.
- Ghazanfar AA, Takahashi DY (2014). The evolution of speech: vision, rhythm, cooperation. Trends Cogn Sci 18(10):543-553. doi:10.1016/j.tics.2014.06.004. PMC page.
- Grijseels DM, Fairbank DA, Miller CT (2024). A model of marmoset monkey vocal turn-taking. Proc R Soc B 291(2026):20240150. doi:10.1098/rspb.2024.0150. Full text.
- Gultekin YB, Hage SR (2017). Limiting parental feedback disrupts vocal development in marmoset monkeys. Nat Commun 8:14046. doi:10.1038/ncomms14046. Abstract.
- Gultekin YB, Hage SR (2018). Limiting parental interaction during vocal development affects acoustic call structure in marmoset monkeys. Sci Adv 4(4):eaar4012. doi:10.1126/sciadv.aar4012. Abstract.
- Katsu N, Yamada K, Okanoya K, Nakamichi M (2019, online 2018). Temporal adjustment of short calls according to a partner during vocal turn-taking in Japanese macaques. Curr Zool 65(1):99-105. doi:10.1093/cz/zoy077. Abstract and PMC page.
- Miller CT, Beck K, Meade B, Wang X (2009). Antiphonal call timing in marmosets is behaviorally significant: interactive playback experiments. J Comp Physiol A 195(8):783-789. doi:10.1007/s00359-009-0456-1. Abstract and PMC page.
- Miller CT, Wang X (2006). Sensory-motor interactions modulate a primate vocal behavior: antiphonal calling in common marmosets. J Comp Physiol A 192(1):27-38. doi:10.1007/s00359-005-0043-z. Abstract.
- Okobi DE, Banerjee A, Matheson AMM, Phelps SM, Long MA (2019). Motor cortical control of vocal interaction in neotropical singing mice. Science 363(6430):983-988. doi:10.1126/science.aau9480. Abstract.
- Pika S, Wilkinson R, Kendrick K, Vernes S (2018). Taking turns: bridging the gap between human and animal communication. Proc R Soc B 285(1880):20180598. doi:10.1098/rspb.2018.0598. Abstract.
- Pomberger T, Risueno-Segovia C, Löschner J, Hage SR (2018). Precise motor control enables rapid flexibility in vocal behavior of marmoset monkeys. Curr Biol 28(5):788-794. doi:10.1016/j.cub.2018.01.070. Abstract.
- Qi R, Lin Y, Liu S, Cao X, Xie M, Yu C, Sun H, Gao L, Li X (2025). Vocal taking turns is premature at birth and improved by the postnatal phonetic environment in marmosets. Natl Sci Rev 12(7):nwaf162. doi:10.1093/nsr/nwaf162. Full text.
- Roy S, Miller CT, Gottsch D, Wang X (2011). Vocal control by the common marmoset in the presence of interfering noise. J Exp Biol 214(21):3619-3629. doi:10.1242/jeb.056101. Abstract and PMC page.
- Takahashi DY, Fenley AR, Ghazanfar AA (2016). Early development of turn-taking with parents shapes vocal acoustics in infant marmoset monkeys. Philos Trans R Soc B 371(1693):20150370. doi:10.1098/rstb.2015.0370. Full text.
- Takahashi DY, Fenley AR, Teramoto Y, Narayanan DZ, Borjon JI, Holmes P, Ghazanfar AA (2015). The developmental dynamics of marmoset monkey vocal production. Science 349(6249):734-738. doi:10.1126/science.aab1058. Abstract.
- Takahashi DY, Liao DA, Ghazanfar AA (2017). Vocal learning via social reinforcement by infant marmoset monkeys. Curr Biol 27(12):1844-1852. doi:10.1016/j.cub.2017.05.004. Abstract.
- Takahashi DY, Narayanan DZ, Ghazanfar AA (2013). Coupled oscillator dynamics of vocal turn-taking in monkeys. Curr Biol 23(21):2162-2168. doi:10.1016/j.cub.2013.09.005. Abstract only; full text could not be opened.

### Chorusing in insects and frogs

- Greenfield MD (1994). Synchronous and alternating choruses in insects and anurans: common mechanisms and diverse functions. Am Zool 34(6):605-615. doi:10.1093/icb/34.6.605. Abstract.
- Greenfield MD, Tourtellot MK, Snedden WA (1997). Precedence effects and the evolution of chorusing. Proc R Soc Lond B 264(1386):1355-1361. doi:10.1098/rspb.1997.0188. Abstract.
- Ravignani A, Verga L, Greenfield MD (2019). Interactive rhythms across species: the evolutionary biology of animal chorusing and turn-taking. Ann N Y Acad Sci 1453(1):12-21. doi:10.1111/nyas.14230. Full text.

### Neural substrates: mammals

- Chen J, Markowitz JE, Lilascharoen V, Taylor S, Sheurpukdi P, Keller JA, Jensen JR, Lim BK, Datta SR, Stowers L (2021). Flexible scaling and persistence of social vocal communication. Nature 593(7857):108-113. doi:10.1038/s41586-021-03403-8. Abstract.
- Eliades SJ, Miller CT (2017). Marmoset vocal communication: behavior and neurobiology. Dev Neurobiol 77(3):286-299. doi:10.1002/dneu.22464. Abstract.
- Eliades SJ, Tsunada J (2018). Auditory cortical activity drives feedback-dependent vocal control in marmosets. Nat Commun 9:2540. doi:10.1038/s41467-018-04961-8. Abstract.
- Eliades SJ, Wang X (2003). Sensory-motor interaction in the primate auditory cortex during self-initiated vocalizations. J Neurophysiol 89(4):2194-2207. doi:10.1152/jn.00627.2002. Abstract.
- Eliades SJ, Wang X (2008). Neural substrates of vocalization feedback monitoring in primate auditory cortex. Nature 453(7198):1102-1106. doi:10.1038/nature06910. Abstract.
- Flinker A, Chang EF, Kirsch HE, Barbaro NM, Crone NE, Knight RT (2010). Single-trial speech suppression of auditory cortex activity in humans. J Neurosci 30(49):16643-16650. doi:10.1523/JNEUROSCI.1809-10.2010. Abstract.
- Gavrilov N, Hage SR, Nieder A (2017). Functional specialization of the primate frontal lobe during cognitive control of vocalizations. Cell Rep 21(9):2393-2406. doi:10.1016/j.celrep.2017.10.107. Abstract.
- Grijseels DM, Prendergast BJ, Gorman JC, Miller CT (2023). The neurobiology of vocal communication in marmosets. Ann N Y Acad Sci 1528(1):13-28. doi:10.1111/nyas.15057. Abstract.
- Hage SR, Jürgens U (2006a). On the role of the pontine brainstem in vocal pattern generation: a telemetric single-unit recording study in the squirrel monkey. J Neurosci 26(26):7105-7115. doi:10.1523/JNEUROSCI.1024-06.2006. Abstract.
- Hage SR, Jürgens U, Ehret G (2006b). Audio-vocal interaction in the pontine brainstem during self-initiated vocalization in the squirrel monkey. Eur J Neurosci 23(12):3297-3308. doi:10.1111/j.1460-9568.2006.04835.x. Abstract.
- Hage SR, Nieder A (2013). Single neurons in monkey prefrontal cortex encode volitional initiation of vocalizations. Nat Commun 4:2409. doi:10.1038/ncomms3409. Abstract.
- Hage SR, Nieder A (2016). Dual neural network model for the evolution of speech and language. Trends Neurosci 39(12):813-829. doi:10.1016/j.tins.2016.10.006. Abstract.
- Holstege G (2014). The periaqueductal gray controls brainstem emotional motor systems including respiration. Prog Brain Res 209:379-405. doi:10.1016/B978-0-444-63274-6.00020-5. Abstract.
- Holstege G, Subramanian HH (2016). Two different motor systems are needed to generate human speech. J Comp Neurol 524(8):1558-1577. doi:10.1002/cne.23898. Abstract.
- Houde JF, Nagarajan SS, Sekihara K, Merzenich MM (2002). Modulation of the auditory cortex during speech: an MEG study. J Cogn Neurosci 14(8):1125-1138. doi:10.1162/089892902760807140. Abstract.
- Jürgens U (2002). Neural pathways underlying vocal control. Neurosci Biobehav Rev 26(2):235-258. doi:10.1016/s0149-7634(01)00068-9. Abstract.
- Jürgens U (2009). The neural control of vocalization in mammals: a review. J Voice 23(1):1-10. doi:10.1016/j.jvoice.2007.07.005. Abstract.
- Jürgens U, Pratt R (1979). Role of the periaqueductal grey in vocal expression of emotion. Brain Res 167(2):367-378. doi:10.1016/0006-8993(79)90830-8. Abstract.
- Jürgens U, von Cramon D (1982). On the role of the anterior cingulate cortex in phonation: a case report. Brain Lang 15(2):234-248. doi:10.1016/0093-934x(82)90058-x. Listing only.
- Luo J, Hage SR, Moss CF (2018). The Lombard effect: from acoustics to neural mechanisms. Trends Neurosci 41(12):938-949. doi:10.1016/j.tins.2018.07.011. Abstract.
- MacLean PD, Newman JD (1988). Role of midline frontolimbic cortex in production of the isolation call of squirrel monkeys. Brain Res 450(1-2):111-123. doi:10.1016/0006-8993(88)91550-8. Abstract.
- Michael V, Goffinet J, Pearson J, Wang F, Tschida K, Mooney R (2020). Circuit and synaptic organization of forebrain-to-midbrain pathways that promote and suppress vocalization. eLife 9:e63493. doi:10.7554/eLife.63493. Abstract.
- Miller CT, Thomas AW, Nummela SU, de la Mothe LA (2015). Responses of primate frontal cortex neurons during natural vocal communication. J Neurophysiol 114(2):1158-1171. doi:10.1152/jn.01003.2014. Abstract.
- Nummela SU, Jovanovic V, de la Mothe L, Miller CT (2017). Social context-dependent activity in marmoset frontal cortex populations during natural conversations. J Neurosci 37(29):7036-7047. doi:10.1523/JNEUROSCI.0702-17.2017. Abstract.
- Risueno-Segovia C, Hage SR (2020). Theta synchronization of phonatory and articulatory systems in marmoset monkey vocal production. Curr Biol 30(21):4276-4283. doi:10.1016/j.cub.2020.08.019. Abstract.
- Roy S, Zhao L, Wang X (2016). Distinct neural activities in premotor cortex during natural vocal behaviors in a New World primate, the common marmoset. J Neurosci 36(48):12168-12179. doi:10.1523/JNEUROSCI.1646-16.2016. Abstract.
- Schneider DM, Nelson A, Mooney R (2014). A synaptic and circuit basis for corollary discharge in the auditory cortex. Nature 513(7517):189-194. doi:10.1038/nature13724. Abstract.
- Sutton D, Larson C, Lindeman RC (1974). Neocortical and limbic lesion effects on primate phonation. Brain Res 71(1):61-75. doi:10.1016/0006-8993(74)90191-7. Listing only.
- Tschida K, Michael V, Takatoh J, Han BX, Zhao S, Sakurai K, Mooney R, Wang F (2019). A specialized neural circuit gates social vocalizations in the mouse. Neuron 103(3):459-472. doi:10.1016/j.neuron.2019.05.025. Abstract.
- Tsunada J, Wang X, Eliades SJ (2024). Multiple processes of vocal sensory-motor interaction in primate auditory cortex. Nat Commun 15:3093. doi:10.1038/s41467-024-47510-2. Abstract.
- Zhao L, Wang X (2023). Frontal cortex activity during the production of diverse social communication calls in marmoset monkeys. Nat Commun 14:6634. doi:10.1038/s41467-023-42052-5. Abstract.

### Neural substrates: songbirds

- Benichov JI, Benezra SE, Vallentin D, Globerson E, Long MA, Tchernichovski O (2016). The forebrain song system mediates predictive call timing in female and male zebra finches. Curr Biol 26(3):309-318. doi:10.1016/j.cub.2015.12.037. Full text.
- Benichov JI, Vallentin D (2020). Inhibition within a premotor circuit controls the timing of vocal turn-taking in zebra finches. Nat Commun 11:221. doi:10.1038/s41467-019-13938-0. Full text.
- Brumm H (2006). Signalling through acoustic windows: nightingales avoid interspecific competition by short-term adjustment of song timing. J Comp Physiol A 192(12):1279-1285. doi:10.1007/s00359-006-0158-x. Abstract.
- Coleman MJ, Day NF, Fortune ES (2022). Neural mechanisms for turn-taking in duetting plain-tailed wrens. Front Neural Circuits 16:970434. doi:10.3389/fncir.2022.970434. Abstract.
- Coleman MJ, Day NF, Rivera-Parra P, Fortune ES (2021). Neurophysiological coordination of duet singing. Proc Natl Acad Sci USA 118(23):e2018188118. doi:10.1073/pnas.2018188118. Abstract and PMC page.
- Fortune ES, Rodríguez C, Li D, Ball GF, Coleman MJ (2011). Neural mechanisms for the coordination of duet singing in wrens. Science 334(6056):666-670. doi:10.1126/science.1209867. Abstract.
- Hamaguchi K, Tschida K, Yoon I, Donald B, Mooney R (2014). Auditory synapses to song premotor neurons are gated off during vocalization in zebra finches. eLife 3:e01833. doi:10.7554/eLife.01833. Abstract.
- Hoffmann S, Trost L, Voigt C, Leitner S, Lemazina A, Sagunsky H, Abels M, Kollmansperger S, Ter Maat A, Gahr M (2019). Duets recorded in the wild reveal that interindividually coordinated motor control enables cooperative behavior. Nat Commun 10:2577. doi:10.1038/s41467-019-10593-3. Abstract.
- Ma S, Ter Maat A, Gahr M (2020). Neurotelemetry reveals putative predictive activity in HVC during call-based vocal communications in zebra finches. J Neurosci 40(32):6219-6227. doi:10.1523/JNEUROSCI.2664-19.2020. Abstract.
- Norton P, Benichov JI, Pexirra M, Schreiber S, Vallentin D (2022). A feedforward inhibitory premotor circuit for auditory-vocal interactions in zebra finches. Proc Natl Acad Sci USA 119(23):e2118448119. doi:10.1073/pnas.2118448119. Full text.
- Schmidt MF, Konishi M (1998). Gating of auditory responses in the vocal control system of awake songbirds. Nat Neurosci 1(6):513-518. doi:10.1038/2232. Abstract.
- Vallentin D, Kosche G, Lipkind D, Long MA (2016). Inhibition protects acquired song segments during vocal learning in zebra finches. Science 351(6270):267-271. doi:10.1126/science.aad3023. Abstract.

### Contingent social feedback, expectation, reward

- Albert RR, Schwade JA, Goldstein MH (2018). The social functions of babbling: acoustic and contextual characteristics that facilitate maternal responsiveness. Dev Sci 21(5):e12641. doi:10.1111/desc.12641. Abstract.
- Bigelow AE, Power M (2016). Effect of maternal responsiveness on young infants' social bidding-like behavior during the Still Face Task. Infant Child Dev 25(3):256-276. doi:10.1002/icd.1974. Abstract.
- Bigelow AE, Power M (2022). Influences of infants' and mothers' contingent vocal responsiveness on young infants' vocal social bids in the Still Face Task. Infant Behav Dev 69:101776. doi:10.1016/j.infbeh.2022.101776. Abstract.
- Bloom K, Esposito A (1975). Social conditioning and its proper control procedures. J Exp Child Psychol 19(2):209-222. doi:10.1016/0022-0965(75)90085-5. Listing only.
- Carouso-Peck S, Goldstein MH (2019). Female social feedback reveals non-imitative mechanisms of vocal learning in zebra finches. Curr Biol 29(4):631-636. doi:10.1016/j.cub.2018.12.026. Abstract.
- Chen Y, Matheson LE, Sakata JT (2016). Mechanisms underlying the social enhancement of vocal learning in songbirds. Proc Natl Acad Sci USA 113(24):6641-6646. doi:10.1073/pnas.1522306113. Abstract.
- Delgado CF, Messinger DS, Yale ME (2002). Infant responses to direction of parental gaze: a comparison of two still-face conditions. Infant Behav Dev 25(3):311-318. doi:10.1016/s0163-6383(02)00096-6. Listing only.
- Diprossimo L, Meyer M, Lüdemann C, Hunnius S, Kärtner J (2026). Infant vocal behavior during contingent vocal imitation and its interruption as a window into the emerging sense of agency. Infancy 31(2):e70080. doi:10.1111/infa.70080. Abstract.
- Elmlinger SL, Schwade JA, Goldstein MH (2019). The ecology of prelinguistic vocal learning: parents simplify the structure of their speech in response to babbling. J Child Lang 46(5):998-1011. doi:10.1017/S0305000919000291. Abstract.
- Elmlinger SL, Schwade JA, Vollmer L, Goldstein MH (2023). Learning how to learn from social feedback: the origins of early vocal development. Dev Sci 26(2):e13296. doi:10.1111/desc.13296. Abstract.
- Gadagkar V, Puzerey PA, Chen R, Baird-Daniel E, Farhang AR, Goldberg JH (2016). Dopamine neurons encode performance error in singing birds. Science 354(6317):1278-1282. doi:10.1126/science.aah6837. Abstract.
- Goldstein MH, King AP, West MJ (2003). Social interaction shapes babbling: testing parallels between birdsong and speech. Proc Natl Acad Sci USA 100(13):8030-8035. doi:10.1073/pnas.1332441100. Abstract.
- Goldstein MH, Schwade JA (2008). Social feedback to infants' babbling facilitates rapid phonological learning. Psychol Sci 19(5):515-523. doi:10.1111/j.1467-9280.2008.02117.x. Abstract.
- Goldstein MH, Schwade JA, Bornstein MH (2009). The value of vocalizing: five-month-old infants associate their own noncry vocalizations with responses from caregivers. Child Dev 80(3):636-644. doi:10.1111/j.1467-8624.2009.01287.x. Full text.
- Gros-Louis J, West MJ, Goldstein MH, King AP (2006). Mothers provide differential feedback to infants' prelinguistic sounds. Int J Behav Dev 30(6):509-516. doi:10.1177/0165025406071914. Abstract.
- Henning A, Striano T (2011). Infant and maternal sensitivity to interpersonal timing. Child Dev 82(3):916-931. doi:10.1111/j.1467-8624.2010.01574.x. Abstract.
- Mesman J, van IJzendoorn MH, Bakermans-Kranenburg MJ (2009). The many faces of the Still-Face Paradigm: a review and meta-analysis. Dev Rev 29(2):120-162. doi:10.1016/j.dr.2009.02.001. Listing only.
- Miller J, Lossia A (2013). Prelinguistic infants' communicative system: role of caregiver social feedback. First Language 33(5):524-544. doi:10.1177/0142723713503147. Abstract.
- Nadel J, Carchon I, Kervella C, Marcelli D, Réserbat-Plantey D (1999). Expectancies for social contingency in 2-month-olds. Dev Sci 2(2):164-173. doi:10.1111/1467-7687.00065. Abstract.
- Rheingold HL, Gewirtz JL, Ross HW (1959). Social conditioning of vocalizations in the infant. J Comp Physiol Psychol 52(1):68-73. doi:10.1037/h0040067. Listing only.
- Striano T, Henning A, Stahl D (2005). Sensitivity to social contingencies between 1 and 3 months of age. Dev Sci 8(6):509-518. doi:10.1111/j.1467-7687.2005.00442.x. Abstract.
- Striano T, Henning A, Stahl D (2006). Sensitivity to interpersonal timing at 3 and 6 months of age. Interaction Studies 7(2):251-271. doi:10.1075/is.7.2.08str. Abstract.
- Syal S, Finlay BL (2011). Thinking outside the cortex: social motivation in the evolution and development of language. Dev Sci 14(2):417-430. doi:10.1111/j.1467-7687.2010.00997.x. Abstract.
- Tanaka M, Sun F, Li Y, Mooney R (2018). A mesocortical dopamine circuit enables the cultural transmission of vocal behaviour. Nature 563(7729):117-120. doi:10.1038/s41586-018-0636-7. Abstract.
- Tronick E, Als H, Adamson L, Wise S, Brazelton TB (1978). The infant's response to entrapment between contradictory messages in face-to-face interaction. J Am Acad Child Psychiatry 17(1):1-13. doi:10.1016/s0002-7138(09)62273-1. Listing only.
- Warlaumont AS, Finnegan MK (2016). Learning to produce syllabic speech sounds via reward-modulated neural plasticity. PLoS One 11(1):e0145096. doi:10.1371/journal.pone.0145096. Abstract.
- Warlaumont AS, Richards JA, Gilkerson J, Oller DK (2014). A social feedback loop for speech development and its reduction in autism. Psychol Sci 25(7):1314-1324. doi:10.1177/0956797614531023. Full text.
- Weisberg P (1963). Social and nonsocial conditioning of infant vocalizations. Child Dev 34:377-388. doi:10.1111/j.1467-8624.1963.tb05145.x. Abstract.

### Looked for and not usable

- Takahashi, Narayanan & Ghazanfar 2013: full text and model details not accessible; see the note at the top.
- Trevarthen 1979 and Murray & Trevarthen 1985 (the classic "protoconversation" and double-video chapters): book chapters, not opened, not used as sources. Nadel 1999 replicates the double-video result and was read.
- Zelick & Narins 1985 (frog call oscillator, J Comp Physiol A 156:223-229, doi:10.1007/bf00610865) and Ghazanfar, Liao & Takahashi 2019 (Anim Behav 151:239-247, doi:10.1016/j.anbehav.2019.01.021): listing only, not used.
- Hsu, Fogel & Messinger 2001 and Molemans 2011: known only through Franklin 2014.
- A table of infant vocalisation rate by month of age, alone versus with caregiver: not found in readable text.
- Phrase and bout durations from Lynch 1995: not in the abstract.
- Timescales in seconds for the meerkat inhibition and excitation rules (Demartsev 2018): not in the abstract.
- Any infant study that measures inhibitory control or a neural pathway against turn-taking: none found.
