# The brain's world: what it sees and hears

Everything the brain lives through is a *moment*: a 64x64 photo (or darkness,
eyes closed), an utterance heard as raw UTF-8 bytes, and what a person would
answer. The brain never sees the kind of a question or its deadline; the
environment uses them only to reward (a reward halves for every tick past the
hidden deadline). Code: `organic/world.py`.

## Sources

| source | what | how answers are vetted | kinds |
|---|---|---|---|
| COCO val2017 annotations | 5,000 photos; person present? how many people? is there a dog? biggest thing? where is the person? | human annotations | presence, count, identity, where |
| COCO train2017 annotations | 118,000 more photos (64x64 thumbnails, memory-mapped), same questions | human annotations | same |
| adults on COCO photos | Nemotron VL writes 5 questions per photo as the brain would see it; Dots 3 answers independently | kept only when both agree | gist, color, count, where, identity, activity, text |
| VQA v2 | questions people asked about the same COCO photos (val2017 and train2017) | answer given by at least 5 of the 10 people asked; kept as typed ("2", not "two") | count, color, where, activity, presence, yesno, ask |
| reading | a word from a list of 48 printed on a photo | rendered by us | text |
| CIFAR-100 | 50,000 small photos: what is this? is it a person? | labels | identity, presence |
| text world (adults) | everyday questions, chat replies, logic and arithmetic, tiny stories; 4 formats rotating over ~80 topics | the other model gives its own answer and judges whether the writer's is also fine | talk, chat, reason, story |
| DailyDialog | something a person said and the other person's reply (replies up to 4 words) | human conversations | chat |
| CommonsenseQA, ARC-Easy | a question heard with its possible answers (answers of up to 3 words) | human-written | reason |
| WinoGrande | a sentence with a gap and the two things that could fill it | human-written, crowd-validated | reason |
| bAbI | a short story (up to 6 sentences) then a question about it | generated tasks with one right answer | story |

Text-only moments are lived with eyes closed (a black image).

## Kinds and hidden deadlines (ticks)

| kind | deadline | why |
|---|---|---|
| chat | 2 | social replies are immediate ("thank you" -> "you're welcome") |
| gist, presence | 2 | a glance is enough |
| talk | 3 | recall of everyday facts |
| color, where | 4 | needs a look, not a search |
| count, identity, activity, text, yesno, ask, reason, story | none | only being right matters |

Moments are drawn kind first (uniformly), then an item of that kind, so
abundant kinds (presence: hundreds of thousands) do not crowd out rare ones.

## Splits and frozen worlds

Test items are those whose photo (or text item) hashes into 10% (crc32 of its
reference), so the split is stable however the world grows. A controlled
experiment can freeze the world: `WORLD_DIR` points to a copy of the adults'
files, and a `sources.json` there lists which other sources are on
(`data/world_ctl/` holds the 2026-10-21 world: COCO val2017 + VQA + bAbI +
CIFAR-100 + the text world of that day).

## Budgets (OpenRouter, half of the account's allowance)

<=10 requests/min, <=500/day: text world 240, photo questions 200, reserve 60.
Jev capped at $1.50. Every reply is cached; nothing is asked twice.

## Design choices made here (environment, not brain)

- Datasets were tokenized by their authors ("you ?"); spacing before
  punctuation is undone so the ear hears text as it would be written. Nothing
  else is changed: no lowercasing of bAbI, numbers in VQA stay digits.
- Only short replies (DailyDialog) and short stories (bAbI) are kept: a child's
  span, and the brain's answer is one word-form.
- Agreement between two adults stands in for "a person would say this".

## 2026-11-04: free-model scan, naming frames

Scan of OpenRouter's free models (`experiments/world/model_scan.py`, `out/model_scan.log`):

| model | writing (10 toddler items) | looking (64px photo) | decision |
|---|---|---|---|
| recursive/flint-v1.0 | 10/10, right | text only | **added** as a writer for text and curriculum items (judged by the others) |
| nvidia/nemotron-3-ultra-550b | 10/10, but "which fruit is yellow?" accepted apple/pear/mango | text only | not yet: wrong answers in writing |
| google/gemma-4-31b / 26b | no reply (busy upstream) | – | retry another day |
| stealth/crouton-alpha | 10/10, right | "red apple" for an apple | not yet: a stealth model (logs prompts, may vanish) |

**Naming frames.** All 7,206 CIFAR picture items in W1 (90% of the first-words world) were
asked as "what is this?". The ear fused that frame into one unit (docs/EAR_SEGMENTATION.md).
`experiments/world/naming_frames.py` has two adults write the ways parents ask a toddler
to name a picture, and a third judge them. `organic/curriculum.py` gives each picture one
of the kept phrasings (fixed per picture). A life's test set is fixed when it starts, so
kid_s3 keeps being tested on "what is this?" until it is restarted; its lived world gets
the variety at its next world refresh.

## 2026-11-05: Wikipedia, Wikidata and Wikimedia Commons (user: "use wikipedia… wiki images also")

Three sources, no model requests, all CC BY-SA / CC0 (Wikipedia, Wikidata and Commons
contributors), used for research:

1. **Wikidata facts** (`experiments/world/wikidata_facts.py` → `data/world/wikidata_facts.jsonl`):
   facts with exact answers for the school stages, only about well-known subjects (many
   Wikipedia editions link them). There are 1,856 facts: capitals, continents and official
   languages of 196 countries, the country of 571 big cities, 565 inventions → inventor,
   33 famous paintings → painter, 161 books → author, mountains and rivers → continent,
   and 367 animal species → class. Answers keep the English name and only those aliases
   that share a word with it (Wikidata's alias lists contain vandalism: "ghahere" for
   Cairo). "Discoverer or inventor" is one property in Wikidata ("who invented
   Australia?"), so inventions are kept only when their description names a made thing.
   Animal classes come from QLever (the same Wikidata on a faster engine; the taxonomy
   walk times out on query.wikidata.org). The class a child would say wins (birds sit
   under reptiles cladistically), and only common names heard in Simple English
   Wikipedia are kept. `organic/curriculum.py` turns them into questions of their stage
   (FACTS: animal class G1 plus "is a whale a fish?" yes/no; continents G2; capitals,
   languages, cities, inventors, painters G3; authors G4). All facts of a stage form one
   source ("wikidata"), so they take one turn among that stage's sources.
2. **Commons picture book** (`experiments/world/commons_pictures.py` → `~/datasets/commons/`):
   for each picture-book word, the members of its kinds that Wikidata links to a photo
   (P18): dog, cat, horse, cattle and rabbit breeds; car, bus, aircraft and locomotive
   models; ship classes; bird, bear, elephant, turtle and mouse species; tiger and lion
   subspecies; apple cultivars; kinds of chairs, beds, cups, clocks and bicycles. Up to
   300 per word, as 64x64 center crops (like CIFAR), fetched at one per second with a
   polite user agent, resumable, memory-mapped. In W1 as "commons:name", asked with the
   adults' naming phrasings.
3. **Simple English Wikipedia as overheard speech** (`experiments/world/simplewiki_speech.py`
   → `~/datasets/simplewiki/sentences.txt`): 2.0M plain sentences of 4–25 words from
   286k articles, lowercased. Not given to any life yet: it changes what the ear learns,
   so it is tested first (`experiments/overheard_probe.py`, docs/EAR_SEGMENTATION.md).

The test split of picture-naming items is decided by the picture, not the wording, so
the CIFAR split is the same as when every item was "what is this?".
