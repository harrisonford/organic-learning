# What the fly connectome says about visual-motor organisation

Data: FlyWire whole-brain connectome, release v783 (139,255 neurons, 54.5
million synapses; Dorkenwald et al. 2024, Schlegel et al. 2024),
[Zenodo record 10676866](https://zenodo.org/records/10676866), with the
[FlyWire annotations](https://github.com/flyconnectome/flywire_annotations).
Script: `experiments/fly/connectome.py`; raw output: [fly_connectome_output.txt](fly_connectome_output.txt).

## Census

| part | neurons | our analogue |
|---|---|---|
| optic lobes | 77,541 (56% of the brain) | the eye organ |
| photoreceptors and other visual sensors | 11,391 | retina |
| visual projection neurons (optic lobe -> central brain) | 8,038 in 326 types, 86% excitatory | organ output |
| visual centrifugal neurons (central brain -> optic lobe) | 524 in 108 types, 57% GABA, plus dopamine and octopamine | feedback ports |
| central brain | 32,383 | orchestrator, memory |
| descending neurons (brain -> body) | 1,303 in 473 types | motor commands |
| brain motor neurons | 110 | motor organ |

## Wiring

| question | finding |
|---|---|
| feedback vs feedforward synapses between brain and eye | 0.59 (1.28 M back vs 2.18 M forward) |
| input to descending neurons | 66% central brain, 11% other descending, 10% ascending (from the body), 6.7% visual projection neurons |
| hops from photoreceptors to descending neurons (edges >= 5 synapses) | mostly 3-4 |
| input to centrifugal (feedback) neurons | 42% central, 27% optic lobe, 20% visual projection, 2% descending |
| neurons driving both descending and centrifugal neurons | 38% of those that drive centrifugal neurons |
| descending neurons with direct visual projection input | 477 of 1303, median 3 visual types each (max 24) |
| strongest direct visual -> descending channels | looming / collision detectors: LPLC4, LPLC2, LC4, LPLC1 |
| LC4 -> giant fibre synapse count vs neuron position | not reproduced (correlations -0.22 to +0.29) with anchor positions; Dombrovski et al. 2023 used dendrite positions in the lobula |

## Insights for this project

1. **Feedback is few neurons with enormous reach.** 524 feedback neurons carry 59%
   as many synapses as 8,038 output neurons, mostly inhibitory or modulatory. The
   brain broadcasts gain and gating rather than steering the eye in detail:
   control ports should be a few broad channels (where, how long, which
   features to boost).
2. **Two loops, and the reflex loop is for danger.** Direct eye -> motor shortcuts
   carry ~7% of descending input and are dominated by looming detectors; two
   thirds of motor drive goes through the central brain. A reflex loop should
   respond to urgent things, not to general salience.
3. **Shallow and wide.** Light reaches motor commands in 3-4 hops despite
   139k neurons; power comes from hundreds of parallel feature types, not long
   chains. Favour many parallel channels over deep stacks of clumps.
4. **Commands are a library of small channels that talk to each other**
   (473 descending types, 11% of their input from other descending neurons).
5. **The eye is told about movement.** Drive to the feedback neurons is
   largely shared with motor commands, consistent with efference copy: the eye
   could receive a "saccade coming" signal.
