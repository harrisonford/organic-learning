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
