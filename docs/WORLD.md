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
   languages of 196 countries, the country of 571 big cities, 115 inventions → inventor,
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

## 2026-11-25: an adult withdrawn by its provider
`dots-studio/dots-3-note-preview:free` is no longer in the provider's public list of models; every request to it
has answered 404 since the night of 2026-11-25 (on 2026-11-24 it still wrote and judged: 201 photo requests, the
last batch 20 questions, 20 judged right). The photo daemon (`experiments/world/annotate.py`) has only two adults,
one writes and the other checks, so every batch failed and it wrote nothing for 31 hours (the health check's
alert at 09:00); the text and curriculum daemons lost the batches where it was the writer or the checker. The
failed requests were counted against the day's allowance (459 of 500 used by 09:20, a good part of it on 404s).
- **Done:** removed from the text and curriculum daemons' lists (three adults left; writer and checker still
  differ), both restarted.
- **The photo daemon** needs a second adult that can look. It is left as it is until one has passed the scan
  (`experiments/world/model_scan.py`: ten toddler items written, one 64-pixel photo named). Free models that take
  pictures today: inclusionai/ling-3.2-flash, particle-ai/particle-1, google/gemma-4-31b-it and 26b,
  nvidia/nemotron-3-nano-omni. The first scan of the two gemma models and ling gave no reply in ten minutes
  (busy upstream, as on 2026-11-04); ling alone is being tried again (`organic-scan1` → out/model_scan.log).
  One adult writing and checking its own questions is not agreement between two adults, so that is not done.
- The one-shot frame scripts (naming, observation, echo frames) still name the withdrawn model; to be changed
  when they are next run.
- **2026-11-25 10:03: the scan's row and the decision.** `inclusionai/ling-3.2-flash:free`: writing 10 of 10 well-formed and
  right in the two samples read ("What does a doggy say?" woof / woof woof / bark), 211 seconds (busy
  upstream); looking: "Apple" for the 64-pixel apple. By the same bar as on 2026-11-04 it passes both jobs, and
  it is a named provider's model, not a stealth one. **Added as the photo world's second adult**
  (`experiments/world/annotate.py`; the daemon restarted; today's allowance is spent, so it writes again after
  midnight). What to watch tomorrow: the share of its questions the other adult judges right, against the
  withdrawn model's; if it is slow the photo world will make fewer than its 200 requests a day. The scan cost
  23 requests of the reserve (retries while the models were busy).
- **2026-11-27 04:52: a spare adult for the photo world.** ling has answered HTTP 429 (the free provider's rate limit)
  to every request since midnight, so no photos were added for 25 hours (10,008 done). The scan today (four
  requests, out/model_scan.log): `particle-ai/particle-1:free` writes 10 of 10 well-formed (dog: woof / bow-wow / arf)
  and names the 64-pixel apple; `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` too (17 s). particle-1 is
  taken as a **spare**: a rate-limited model rests 30 minutes and the spare takes its place in the pair meanwhile,
  so two different adults still write and check (another maker than the nemotron adult, so the check is not a
  model family agreeing with itself). With both adults answering, nothing changes (`annotate.py`, `SPARE`).


## 2026-11-25: more Wikimedia Commons photographs, fetched constantly (not part of world 4)

Kris, 22:20: "yes slow polite download constantly", after "we should have also wiki pics". New service
`organic-commons` (`experiments/world/commons_depicts.py`, unit file in ~/.config/systemd/user, enabled, restarts
by itself): Commons files that the site's editors have marked as depicting a thing (structured data, P180),
found through the site's search a page of 25 at a time, word after word.

- **Kept:** JPEG or PNG whose 120-pixel thumbnail is at least 64 pixels on its short side; the middle square at
  64 × 64 (never enlarged); the word, the file's title, its page id. The title is what a licence and an author
  are looked up by; the pictures are not passed on to anyone.
- **Politeness:** one request at a time, a picture every two seconds (about 40,000 a day at most), the earlier
  fetcher's user agent, `maxlag` on the API, five minutes' wait (or what the server says) on a refusal.
- **Where:** `~/datasets/commons_more/`. **Not** `~/datasets/commons`, which the lives' world reads and which
  must not change while lives are in it. Nothing reads the new directory yet.
- **Words:** 39 (the photo benchmark's and the picture book's, plus flower, tree, house). Each word's Wikidata
  item is checked against its English label first; giraffe was left out (the item I had is the northern
  giraffe). The search gives at most 10,000 files a word.
- **What "depicts" means, seen in the first three:** the thing is somewhere in the picture, not always its
  subject, and paintings count (one of the first three dogs is in a painting of 1899, one is small in a
  courtyard). Good for "all kinds of pictures" and for development without labels; as labelled pictures they
  are noisier than COCO's crops, to be measured before they are used as tests.
- Log: out/commons_depicts.log. Not in health.py's list yet.

## 2026-11-26: Wikimedia Commons video, fetched constantly (queue E 31; not part of world 4)

Kris, 08:30: "we can also download in parallel wikimedia then, this pc has 1 TB disk". New service
`organic-commons-video` (`experiments/world/commons_video.py`; unit in ~/.config/systemd/user, enabled, restarts by
itself; log out/commons_video.log). Commons has about 420,000 videos (search, 2026-11-26): about 21,000 under
"Videos of animals", 7,700 vehicles, 35,500 people, 140,000 nature (with subcategories).

- **Searches, in turn:** videos marked as depicting one of 39 picture-book words (few: 67 for dog), then videos under
  the four subject categories (no word, the category kept). The search reaches at most 10,000 results each.
- **Kept:** the smallest served version at least 160 pixels on its short side (usually a 240p VP9 transcode, about
  3 MB a minute), stored whole in `raw/` so other sizes and rates can be taken later; and its first 120 seconds at 10
  frames a second, the middle square at 64 × 64, appended in order to `frames_64.u8` with one line per video in
  `videos.jsonl` (first frame, count, title for the licence and author, page id, word or category, duration).
  Skipped: under 2 s or over 20 minutes; no small version and the original over 50 MB.
- **Politeness:** one request at a time, 3 s between requests, the same user agent, `maxlag`, five minutes' wait on
  a refusal. It runs beside the picture fetcher (one picture every 2 s); together about one request a second.
- **Where:** `~/datasets/commons_video/` (its own directory; ffmpeg decodes from there).
- **Seen in the check:** one of the first two "dog" videos is a CGI cartoon (a dog in an animated short): Commons'
  "depicts" includes animation. Kept, and to be counted before the frames are used: real footage and animation
  will be told apart by category when the set is read.
- **Rates:** a frame changes by 1.5 grey levels on average from one tenth of a second to the next in the beach
  video and by 10.6 over five seconds; 6.2 and 44.7 in the cartoon (cuts).

Beside it, `organic-core50-fetch`: the handheld-object videos (CC BY 4.0, 5.9 GB) at the server's pace, about 180 KB/s
(about nine hours).

## 2026-11-26: SpokenCOCO, people reading the COCO captions aloud (queue E 32; not part of world 4)

For the ear that hears sound and is never given text (Kris, 2026-11-26 12:50; queue E 32 in docs/LOOP.md).
SpokenCOCO (Hsu, Harwath, Song & Glass, NeurIPS 2020 workshop on self-supervised learning for speech): about
600,000 recordings of people reading the MSCOCO captions out loud, each caption once; audio only, the images are
COCO's own (we hold the 2017 train images as 64 × 64 thumbnails). Licence: Creative Commons Attribution-ShareAlike,
as stated for all the corpora on https://sls.csail.mit.edu/downloads/placesaudio/ (checked 2026-11-26). Anything made
public from it must attribute and share alike; the speakers' voices are theirs: never published by us.
`~/datasets/spokencoco/fetch.sh` (unit `organic-spokencoco-fetch`; curl limited to 3 MB/s, resumable, generic
user agent; 64 GB, about six hours) → `SpokenCOCO.tar.gz`; done when `tar` is in `fetch.done`. Downloaded data:
read with `python -I`, never run. The captions' text, if the archive holds it, is for measuring only (which word was
said where): the brain never gets it.
