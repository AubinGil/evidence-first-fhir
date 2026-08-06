# OCR is a trust boundary, not a function call

Recorded 2026-08-02, from a live audit of the local and AWS ingestion paths,
then extended the same day with measurements against generated ground truth.

Everything below was measured with the **`v2_english` checkpoint** of Nemotron
OCR v2 (53.8M). The multilingual checkpoint (83.9M) is untested here, and the
model card reports an order-of-magnitude accuracy difference off-language, so
none of these numbers transfer to non-English documents.

This note covers transcription fidelity. `measurement-log.md` covers everything
downstream of it -- extraction models, the grounding gate, and retrieval.

**Where the code is.** The pipeline this note reasons about is in this
repository -- `discharge_to_fhir.py`, the schemas, the policies. The
measurement harness is not. The reader registry, the two scorers, the routing
implementation and the fixture generator all require three running OCR engines
and a GPU, so they stay in the private lab this repository was extracted from,
and files cited below that you cannot find here are that harness. The numbers
are therefore reported rather than reproducible from this repository alone.

**How to read the engine comparisons.** The corpus is synthetic discharge
summaries rendered in one typeface from one generator, one seed, one language,
and each engine is reached through one serving path and one prompt. That is
enough to characterise *failure modes*, which is what this note is for. It is
not enough to rank products. Where a comparison appears to name a winner it
means "on this corpus, in this configuration" -- the numbers are reported so
that scope travels with them, not as a general verdict on any engine.

## The finding

`grounded_facts` accepts a fact only when its `evidence_quote` is an exact
substring of the note, then records character offsets into that note
(`discharge_to_fhir.py:79`, `:88`). The check is correct. The **anchor** is the
problem: when the note was transcribed, those offsets locate the claim in the
transcript, not in the document.

The gate can therefore certify only *"supported by our transcription."* The
project's claim is *"supported by the document."* Those diverge exactly when OCR
is wrong, and nothing in the original design could detect the difference.

This is a scope gap, not a logic bug. Grounding is only ever as strong as the
fidelity of the text it grounds against — exact for pasted text, probabilistic
for anything transcribed.

## Why it is dangerous rather than merely imperfect

Both engines tested produced errors that are **plausible, not garbled**:

| Engine | Source | Transcribed |
|---|---|---|
| AWS Textract | `A1c is intentionally older...` | `Alc is intentionally older...` |
| Nemotron OCR | `Type 2 diabetes mellitus` | `Type diabetes mellitus` |
| Nemotron OCR | `1968-02-14 / 58 years` | `1968-02-14/ / 58 years` |

A fact quoting `Alc` grounds cleanly against a transcript containing `Alc`. The
gate passes, the offsets are precise, and the value is wrong. Garbled output
would be caught; plausible output is not. Textract also gave its *lowest*
confidence on the most clinically important token on the page — `8.2%` at
68.79%, against a page mean of 98.65%.

## OCR is six stages, not one

The AWS reference diagram has no OCR box: step 3 de-identifies and step 4
extracts, implying text arrives ready. Reality, established by making the
fallback route work:

1. **Format conversion** — Nemotron decodes PNG/JPEG only; PDFs returned HTTP
   500 until rasterized. Textract splits further: sync `DetectDocumentText` for
   images, async `StartDocumentTextDetection` for multi-page PDF.
2. **Rasterization DPI** — a free parameter that materially changes accuracy and
   was previously unrecorded. Now `raster_scale` on the transcription record.
3. **Region assembly** — engines return positioned regions, not a document.
4. **Reading order** — regions sharing a row differ in y by ~2e-4; sorting on y
   before grouping orders a row by noise instead of left-to-right.
5. **Confidence propagation** — per-region scores were discarded at the boundary.
6. **Geometry mapping** — coordinates must be mapped back to facts, or evidence
   cannot be shown to a reviewer as anything but an offset.

Treating this as "call OCR, get text" is the assumption the diagram encodes.

## Route states as found

| Route | PDF | Image | Geometry | Usable note |
|---|---|---|---|---|
| Document service (:8787) | 2997 chars, keeps table pipes | — | none | yes |
| Nemotron (fallback) | HTTP 500 | 40 regions, mean conf 0.9365 | present, undetected | **empty** |

The fallback had never worked: Nemotron returns `{regions, count}` with no
`text` key, so `ocr["text"]` was always `""` and `process_document` raised
*"both document indexing and direct OCR returned no text"* whenever it was
reached. The branch was dead in practice — which is also why nobody noticed it
runs a different engine than the primary route.

Net: no route produced re-anchorable evidence. The working one had no geometry;
the one with geometry produced no text.

## Coordinate convention trap

Nemotron reports `left / upper / right / lower` as normalized scalars. These are
the **numeric bounds of the interval, not visual edges**: `upper` is the larger
y value, i.e. the *bottom* of the box in a top-down frame, and `lower` its top
(height = `upper - lower` ~ 0.010-0.015, one text line). Reading them as
top/bottom flips every box vertically. Mapped explicitly in `scalar_bbox()`.

## Design response

Implemented in `clinical_orchestration.py` (commits a6808a2, 07b237f, a719fcd,
74d918f):

- **Engine recorded per case** — engine, model, endpoint, route, raster scale,
  page count. Which engine ran previously depended on whether :8787 happened to
  be up, and was recoverable only from prose in `degraded[]`.
- **Provenance assessed per case** — `anchor` (source_text | transcript),
  `strength` (exact | re_anchorable | degraded), and the specific `gaps`.
- **Facts re-anchored to geometry** — accepted facts carry `evidence_regions`
  (page + box) and `evidence_confidence_min`, so a fact resting on a weak read
  is visible as such at review time.
- **Pasted text recorded explicitly**, so an exact anchor is distinguishable
  from a transcript anchor rather than implied by absence.

## Consequence for the cloud architecture

A PDF branch inserts OCR *before* the de-identification gate, so raw
pre-de-identified documents are read by another service. That is a change to the
trust boundary, not just an added step, and belongs in the threat model before
any Textract path ships.

## Measured behaviour

Against generated ground truth, 24 fixtures (8 alterations x 3 severities),
1091 scored regions.

**The engine is deterministic.** Repeat runs on identical input produce
byte-identical regions. Every error below is reproducible, which is what makes
them separable rather than a diffuse "OCR is unreliable" risk.

**Errors split into two populations.** Raising rasterization scale recovers
small isolated digits -- at scale 1.0 the case id `929882` was misread, by 2.0
it was correct -- but a recognizer floor persists identically at 1x, 2x, 3x and
4x: `amoxicillin -> amoxicilin`, `clavulanate -> lavvlanate`, `consistent ->
consiiseet`. More pixels do not touch it. Across six fixtures scale 4.0 also
traded fewer missing tokens for *more invented* ones, so the default stays at
2.0.

**Numbers are lost disproportionately.** On a clean render, 7 of 12 lost
tokens were numeric, against numbers being roughly 5-10% of the document. The
same `Type 2 -> Type` deletion reproduced on a second, unrelated document.
Note the mechanism is *not* language-model omission: Nemotron OCR v2 is a
detector-recognizer (RegNetX-8GF detector, transformer recognizer, relational
layout model), not a VLM. The evidence points at detection and segmentation of
small isolated glyphs -- `PAGE 1 OF 2 -> PAGE OF`, `875 mg/125 mg -> 875 mg/1
25 mg` is a boundary split mid-token.

**Confidence discriminates but is not a probability.** AUC 0.733, so faithful
regions do outrank corrupt ones. But ECE 0.214 and MCE 0.503, every bin
overconfident: at a mean confidence of 0.89 the observed accuracy is 0.39. The
distribution is also degenerate -- four of five quantile bins span 0.9329 to
0.9441, a band 0.011 wide holding 80% of regions. That is why the threshold
sweep moves precision only from 0.715 to 0.768 across its entire range, and why
no threshold reaches 0.99. Rescaling would make the numbers honest; it cannot
create separation that is not there.

**Two readers disagree exactly where it matters.** Across the 24 fixtures
neither Nemotron nor GLM-OCR ever corroborated the other's invented tokens --
0 of 355 -- while both lost the same 73 digits. So disagreement catches the
dangerous class and misses the safe one. GLM is unsuitable as a primary
transcript (904 missing tokens against Nemotron's 435) and its failures are
bimodal: zero invented on most fixtures, then 94 and 92 on two. Its value is
being *independently* wrong, not being good.

## Error is concentrated, not distributed

Generator 1.2.0 emits per-region ground truth -- box, reading order, clinical
entity categories, and post-alteration visible fraction -- which allows scoring
by content rather than by token bag. Nine fixtures, 468 regions:

| category | n | CER |
|---|---|---|
| document_identifier | 36 | **0.2711** |
| provenance | 9 | 0.1324 |
| medication | 108 | **0.0660** |
| diagnosis | 18 | 0.0480 |
| care_instruction | 36 | 0.0389 |
| patient_demographics | 72 | 0.0275 |
| vital_sign | 18 | 0.0072 |
| hospital_course | 27 | 0.0011 |
| document_header | 36 | 0.0000 |

Overall CER 3.17%, 4 regions missed of 468, reading-order agreement 0.98.

The engine is good, and its error concentrates in two places: **identifiers**
at roughly 35x the rate of prose, and **medication names**. Both are
low-predictability character sequences. A transformer recognizer transcribes
with contextual support from surrounding characters; prose supplies that,
`SYN-DIS-929882` and `clavulanate` do not. This is the same mechanism behind
the isolated-digit loss.

That suggests a targeted policy rather than a general warning: verify
identifiers and drug names first. It does not license trusting narrative
blindly -- prose scored well here, on rendered Arial from one generator, and
the handwriting section below is where that comfort stops.

Three artifacts had to be removed before these numbers meant anything, and each
had inverted a conclusion. `reading_order` restarts per page, so pooling across
pages scored correct page-2 ordering as discordant (0.61 -> 0.98). Exact-region
match is length-biased, zeroing a 105-character paragraph for one wrong
character. And the engine merges at paragraph level while the renderer records
one region per drawn line, so 1:1 pairing scored surplus lines as never read
(missed 77 -> 4, CER 0.0966 -> 0.0317). `vital_sign` moved from an alarming
0.4544 to 0.0072 on that last fix alone.

## The design, and what it costs

Routing on disagreement, scored against ground truth over 270 regions:

    precision 0.810   of flagged spans, fraction containing a real error
    recall    0.953   of errors, fraction caught
    burden    0.370   fraction of regions flagged
    missed        4   both readers agreed and both were wrong

Precision is **1.00 on medications** (21 of 21), diagnoses, vitals, symptoms
and care instructions. Nearly every false alarm comes from boilerplate:
`document_header` was flagged 8 times and caught nothing, because the second
reader systematically skips repeated page headers.

- **Disagreement decides what enters review.** No threshold required, which
  matters given the ceiling above.
- **Confidence orders the queue.** It separates the bottom 20% well (0.39
  accuracy against 0.70-0.85 above), so it is triage, not a gate.
- **Boilerplate suppression uses cross-page repetition, not entity labels.**
  The labels come from the fixture generator and do not exist at runtime;
  using them would leak the fixture into production. Repetition separates the
  same regions -- headers and markers repeat on every page at 100%, every
  clinical category at 0% -- and is computable from any document. It is off by
  default, because repeated regions include the case identifier and identifiers
  are the highest-error category measured.
- **Absence of a second reader is recorded structurally.** A case reviewed
  without corroboration rests on one reader, and that must be visible at review
  time rather than indistinguishable from agreement.
- **Never show raw confidence to a clinician as a probability.** 0.89 meaning
  39% manufactures false assurance.

Implemented in `clinical_orchestration.route_by_disagreement`, pinned by
`test_disagreement_routing.py`.

## A third reader: DeepSeek-OCR, measured

Recorded 2026-08-04. `deepseek-ocr:latest` served by Ollama on the workstation
GPU, 24 fixtures, seed 20260802, raster scale 2.0, generator 1.3.0. Every
reader is handed the *same* rendered pages, because scale changes what an
engine reads and a comparison where each engine rasterises for itself measures
the renderers too.

The corpus did not move under the earlier numbers: the geometric scorer gives
the incumbent **CER 0.0318** on generator 1.3.0 against **0.0317** published on
1.2.0, so the two generations are comparable.

On this corpus and through this serving path, it did not read well enough to be
a primary candidate, and it paired worse with the incumbent than the second
reader already deployed. "Two things could not be measured" below bounds that
claim: the grounding path that would give it geometry was never run here.

| reader | missing | invented | seconds | geometry |
|---|---|---|---|---|
| Nemotron OCR v2 (`v2_english`) | **458** | **176** | **28.4** | 1092 regions |
| GLM-OCR | 910 | 194 | 84.9 | none |
| DeepSeek-OCR | 2874 | 305 | 63.3 | none |

6.3x the incumbent's missing tokens and 1.7x its inventions -- both error
classes moving together, where a trade between them is the more common pattern.
Whether that is the model or this serving path is exactly what the untraced
early termination below leaves open.

### The failure mode is early termination that reads as success

DeepSeek does not degrade on a degraded page. It stops. On 12 of 24 fixtures it
loses 150 or more tokens of roughly 450, and **those runs are the fast ones**:
2.20s mean against 3.08s for the fixtures it reads properly. Effort falls as
difficulty rises, which is the signature of giving up rather than struggling.

On `blur/medium` it returned **190 characters of 1958** -- the MRN block, a
footer, and the literal token `None` -- in 1.26s, with no error and no warning.
The incumbent read the same page in 1.2s and lost 12 tokens.

This is the pattern the rest of this work keeps finding: a failure converted
into something that reads as success and passed downstream. A 190-character
transcript is a well-formed transcript. Nothing in the pipeline would report
it, the grounding gate would see a short document and quarantine nothing, and
review would show a handful of facts rather than an obviously broken read.

### What a second reader actually buys

| pair | inventions corroborated | caught by disagreement | digits both lost |
|---|---|---|---|
| Nemotron + GLM | **0** of 370 | **100%** | 82 |
| Nemotron + DeepSeek | 2 of 479 | 99.6% | 100 |
| GLM + DeepSeek | 1 of 498 | 99.8% | 137 |

Here, adding DeepSeek to the incumbent measured worse than the deployed pair on
both axes: it corroborated two inventions GLM flagged, and lost 100 digits
jointly with Nemotron against GLM's 82. Correlated loss is the class no
disagreement rule can catch, so that column is the one to watch -- though 100
against 82 on 24 fixtures is a narrow margin, and not one to defend hard.

Across all three readers, 675 inventions, **none made by all three**. The
independence result holds and is not the problem.

### Prompt choice moved a whole error category

`Free OCR.` silently dropped the document identifier (`SYN-DIS-929882`), the
page number, and the synthetic banner on a *clean* fixture -- the small
isolated strings that are already the incumbent's weakest category. The
documented markdown prompt kept all three. A third prompt collapsed the header
into an HTML table and returned a third of the text.

So "DeepSeek-OCR scores X" is not a well-formed statement. The prompt is
recorded in `output/ocr-calibration/readers.json` alongside every number, and
the comparison uses the prompt most favourable to it.

### Two things could not be measured, and were not estimated

A vision language model returns a transcript. It has no boxes and no
confidence, so **region-level scoring and confidence calibration cannot run
against it at all** -- not badly, but not at all. The AUC/ECE/MCE work and the
reading-order agreement have no DeepSeek column and cannot be given one from
Ollama. Grounding mode does emit boxes, but labelled by element type
(`title[[67, 208, 310, 222]]`) with no text attached, and it degraded the
transcript; boxes and text do not come out together. The HF/vLLM path would
give real grounding output and needs a CUDA build neither local environment
has.

`tools/score_ocr_categories.py` closes part of that gap by locating a region in
the transcript by content instead of position, which needs no geometry:

| category | n | Nemotron | GLM | DeepSeek |
|---|---|---|---|---|
| document_identifier | 88 | 0.1600 | 0.5636 | 0.4136 |
| diagnosis | 47 | 0.1548 | **0.0086** | 0.3238 |
| medication | 272 | 0.0519 | **0.0004** | 0.2196 |
| care_instruction | 96 | 0.0738 | **0.0000** | 0.1907 |
| hospital_course | 72 | 0.0378 | 0.0126 | 0.3261 |
| synthetic_marker | 48 | 0.0154 | 0.5589 | 0.3850 |
| patient_demographics | 188 | 0.0041 | 0.0342 | 0.1165 |
| section_header | 166 | 0.0038 | **0.0000** | 0.2109 |
| **overall** | **1239** | **0.0486** | 0.0604 | **0.2689** |

**This metric is not interchangeable with the geometric one.** Run against the
same engine, same fixtures, same generator, the two disagree by 2x overall
(0.0318 geometric against 0.0636 aligned) and by up to ±0.16 per category, and
they reorder the worst-five list. The geometric scorer is authoritative where
it applies, because it groups wrapped lines under the detected paragraph and
cannot be fooled by a repeated string. Use the table above to rank engines
against each other -- the 5.5x DeepSeek/Nemotron gap dwarfs the 2x method
disagreement -- and do not quote a cell from it beside a published geometric
number.

The more interesting result in that table is not DeepSeek. **GLM reads clinical
prose better than the incumbent and identifiers far worse**: medications at
0.0004 against 0.0519, diagnoses at 0.0086 against 0.1548, while losing more
than half of every document identifier and synthetic marker. That is the
incumbent's worst clinical category being read near-perfectly by the engine
already deployed beside it, and it is direct evidence for the standing
suggestion below that running both and taking text from one and geometry from
the other beats choosing. It rests on the weaker metric and should be confirmed
before it is acted on.

### Handwriting inverts the ranking, qualitatively

Two real handwritten clinical documents, 2026-08-04: a cramped prescription and
a legible hospital letter. **n=2, no established ground truth** -- parts of the
first are genuinely ambiguous to a human reader, which is itself the finding.
Not a benchmark; a look. Neither image is stored here, and patient identifiers
are redacted from the examples below.

Every fixture in this corpus is rendered in Arial. Nothing here has ever been
measured on handwriting, and the three engines fail on it in three different
ways:

| reader | shape of failure | seconds |
|---|---|---|
| Nemotron | non-words: `Kioguur Jony tub`, `(Prsscet and`, `$34522` | 0.5 |
| GLM | fluent, plausible, wrong -- then loops for 30456 chars | **74.9** |
| DeepSeek | truncates, then narrates the image instead of reading it | 0.9 |

The incumbent, best on print, degrades to obvious gibberish -- which **fails
safe**: no reviewer or downstream model mistakes `Kioguur Jony tub` for a drug.

GLM inverts. It returns clean, well-formatted, confident text in which
`Biogesic 500mg tab` became **`Biotic 500mg tab`** -- a different plausible
drug-like name -- and `as needed` became **`an hour`**, turning a PRN
instruction into an hourly one. Both are fluent, both are real spans in the
transcript, so both would ground successfully against a document that never
said them. This is the invented-token class the trust boundary exists to catch,
in its worst form. It also ran 20x its normal page latency and emitted the same
block fifteen times, the same runaway shape as `qwen3.5:9b` in the extraction
work.

DeepSeek put `The handwriting is somewhat difficult to read due to the cursive
style, but it includes the following information:` **inside the transcript**,
where it would be indexed, retrieved, and quotable as clinical text. Its
reading of the drug was `Physique 30 mg ptb` -- wrong name and wrong dose.

**No vendor claims this capability.** Checked against primary documentation,
2026-08-04, not secondary write-ups:

| engine | what the official documentation says about handwriting |
|---|---|
| Nemotron OCR v2 | named once, as a *training data* ingredient ("handwritten document pages", ~680K real-world images). No capability claim, no handwriting benchmark, no number. Limitations hedge on "highly stylized fonts". |
| GLM-OCR | no mention, on either the model card or the repository |
| DeepSeek-OCR | no mention, in either the repository or the paper (arXiv 2510.18234) |

The widely repeated "92%+ accuracy on handwritten notes" for DeepSeek traces to
SEO content farms, not to DeepSeek. Both VLMs are benchmarked on OmniDocBench
and Fox, which are printed-document benchmarks -- and the model ranked **#1 on
OmniDocBench V1.5 (94.62)** is the one that produced `Biotic 500mg` and `an
hour` above. Printed-document rank predicts nothing here.

So the measurement is not a contradiction of the vendors. It is a measurement of
exactly the thing all three decline to promise, which is the argument for a
detect-and-refuse gate being a reasonable default here rather than an overly
cautious one.

### A second sample, legible, and the conclusion changes

The prescription above is a hard case. A neat handwritten hospital letter is
not, and it separates two things the first sample confounded. Patient
identifiers are redacted below; the source is not in this repository.

The letter carries a **printed letterhead above handwritten prose**, so it
contains its own control. Nemotron read the letterhead near-perfectly
(`.com` → `.coo`) and the handwriting as non-words (`Ocor Halam/ Bir`,
`gratchul`, `Mcunt t cinle`). Same image, same pass: the variable is isolated.

GLM read the legible hand **almost perfectly** -- "We are grateful for the
support given to our ... patient ... for her treatment. She has a good
prognosis, and is under Maintenance." So the blanket claim that handwriting is
unreadable was wrong. Legibility dominates, and on a clear hand a VLM is
largely correct.

What survives is worse than being wrong everywhere:

| truth | GLM | DeepSeek | class |
|---|---|---|---|
| `Onco patient` | **`Once patient`** | **`once patient`** | the only clinical fact in the letter |
| `Maintaince` | `Maintenance` | `maintenance` | silently corrected the document |
| patient given name | mangled | differently mangled | rare proper noun |
| `Tel: +91 80 2206 5000` | **`2006 5000`** | dropped | a *printed* digit |

**Both VLMs independently made the same error on the one word that mattered.**
`Onco` is a domain abbreviation; `once` is a common English word. A language
prior pulls rare domain tokens toward frequent ones, so two different models
land on the same wrong answer -- and an oncology patient becomes nothing at
all, fluently, in a sentence that reads perfectly.

That is the case disagreement routing cannot see. The comfort from printed text
-- 0 of 355 inventions corroborated -- does not carry over: here the readers
corroborate precisely where they are wrong, because they share the prior that
caused the error. Independence was never a property of the engines; it was a
property of *printed* failures being random. Handwriting failures are
systematic, so they align.

GLM also normalised a misspelling in the source (`Maintaince` → `Maintenance`).
A gate that requires exact spans cannot quote a document the reader silently
corrected, and the corrected form looks more right, not less.

So handwriting is not a problem because it is illegible. It is a problem
because **when it is legible the errors concentrate on exactly the rare domain
terms and identifiers that carry the clinical content, and the readers agree on
them.** Nothing in the pipeline detects this: a page like the letter is
transcribed and a fluent, corroborated, wrong clinical fact goes downstream at
ordinary confidence.

### Scope

**This pipeline is not for handwritten clinical documents.** That is a declared
limit, not a defect to be fixed later. Handwritten prescriptions in particular
are out of scope: the sample above shows a drug name and a dosing instruction
both silently replaced with plausible alternatives, which is the exact failure
the grounding gate cannot catch because the substituted text is a real span.

Independently corroborated: Carchiolo et al., WEBIST 2025, ran Azure Document
Intelligence with a **custom model trained on the target layout** over real
Italian hospital records and measured **average WER 37.43% and CER 14.27%**,
with some documents above 60% WER. In one case the model located every target
field correctly and still returned WER 66.67% -- detection is not the hard part.
Their recurring failure was "confusion in domain-specific terminology", which is
the `Onco` → `Once` result arrived at from a different direction.

The limit is recorded rather than engineered around because every available
mitigation is weaker than the statement. Confidence is badly calibrated on
printed text already (ECE 0.214) and untested off-distribution. Disagreement
routing demonstrably fails here, since the readers agree where they are wrong.
And no vendor claims the capability, so there is nothing to hold to account.
Declaring the boundary is the honest response; a detector would be a way of
holding the boundary automatically, and is only worth building if these
documents ever have to enter the system at all.

GLM looped on both samples -- 30456 and 13885 characters, 74.9s and 32.1s
against ~3.5s for a printed page. Runaway generation on handwriting is
reproducible and is an operational risk on the shadow path.

## Open

- ~~`crop` and `occlusion` produced identical results at all three
  severities~~ -- settled 2026-08-04. On generator 1.3.0 both render three
  distinct PDFs, so that pair is no longer degenerate. **`clean` is**, and
  always was: it is byte-identical at mild, medium and severe, because it has
  nothing to vary. The corpus is 22 distinct fixtures, not 24, and every pooled
  average in this note weights the clean case three times. Re-pooling the
  DeepSeek comparison over the 22 distinct fixtures moves the incumbent from
  19.1 to 19.4 mean missing tokens and DeepSeek from 119.8 to 127.9 -- the
  duplicate flattered DeepSeek, since clean was among its better cases, so the
  conclusion holds and widens slightly.
- GLM's advantage on medications and diagnoses is measured only by the
  geometry-free scorer, which disagrees with the geometric one by 2x. Confirming
  it needs either boxes from GLM or a per-region check by hand.
- Handwriting is closed as a scope limit, not an open thread. The evidence is
  n=2 and qualitative, which is enough to decline the use case and would not be
  enough to accept it. Should that ever need revisiting, the candidate corpora
  are RxHandBD (5578 cropped prescription words), the Kaggle doctor-handwriting
  set (90 samples, 30 prescribers), and IAM for general English -- all word- or
  line-level, none a full clinical page in English.
- DeepSeek's early termination was not traced to a cause. Context length,
  Ollama's image preprocessing, and the model's own resolution modes are all
  untested; the 8192 context it loads with is the first thing to rule out.
- Language and script mixing are unevaluable on `v2_english`.
- The token-multiset comparison in `compare_ocr_readers.py` is superseded by
  region-level scoring for everything except the invented-token count, which is
  still the cleanest evidence that the readers fail independently -- and is now
  the only comparison that works across all three, since two of them emit no
  geometry. It takes `--readers` and generalises the two-reader intersection to
  any number without changing the two-reader definition; the counts above are a
  fresh run on generator 1.3.0, not the 1.2.0 figures earlier in this note.
- Every number here is one document type (discharge summary), one seed, and one
  second reader. The routing precision in particular rests on GLM-OCR's
  specific failure profile, which is bimodal -- near-silent on most fixtures,
  then ~90 invented tokens on two of 24.
- Calibration on the goldset. The synthetic PDF generator
  (`clinical_fhir/synthetic_data/discharge_generator.py`) renders fixtures from
  known text, giving exact ground truth for character-level error rates and a
  defensible confidence threshold for quarantine.
- Two-reader comparison. `glm_ocr_shadow_service.py` already runs a second
  engine, refused at `clinical_orchestration.py:370` as a "parity candidate".
  Disagreement between independent readers is a quarantine signal of the same
  kind as the rest of the pipeline.
- Primary route returns no geometry, so every case through it scores
  `degraded`. Running both engines — text and retrieval from one, geometry and
  confidence from the other — is likely better than choosing.
