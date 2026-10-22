# ForgeSight — 5-minute demo video runbook

Target 4:50–5:00. Record 1920x1080, 30 fps, voiceover over screen capture.
**Every click below was executed against the current build on 2026-10-22** and the
expected on-screen values are the ones that run produces. If you get something
else, stop and check `/health` before re-rolling.

The rules ask the video to show **the team, the application working, its
architecture, and its principal results**. The first and third were missing from
the earlier script; they are beats 1 and 6 now.

The single most important decision: **lead with the closed loop, not the metrics.**
Agentic Vision is judged on whether visual evidence changes a later action. The
metrics section is proof, not the hook.

---

## Before you touch the record button

```powershell
cd C:\Users\wecal\Documents\Codes\opencv-visionguard
aws login                                    # opens a browser; wait for success
aws sts get-caller-identity                  # must print an account id
python eval\smoke_test.py                    # must print 26/26 checks passed
python aws\deploy_endpoint.py up --ttl-hours 3
python aws\deploy_endpoint.py status         # prints http://<IP>
curl.exe http://<IP>/health                  # must show "opencv": "5.0.0"
```

- `up` ships the current `backend/` and `frontend/`. If it says
  `already up: ...`, the old build is still running: `python aws\deploy_endpoint.py down`,
  then `up` again. The "why:" line in beat 5 only exists in the new frontend.
- Budget: 3 hours is about **$0.12** at $0.03988/hr.
- `curl.exe`, not `curl`: PowerShell aliases `curl` to `Invoke-WebRequest`.

### Stop your username appearing on screen

In the PowerShell window you will record, run this once:

```powershell
function prompt { "PS> " }
cls
```

The prompt becomes `PS>` and no `C:\Users\wecal\...` path is visible. Then run
the `curl.exe` command fresh when the camera is rolling.

### Record in this order, not in edit order

The endpoint is the only fragile dependency. Capture it first.

1. Loop (beat 4) — the main event, most likely to need a re-take
2. Veto (beat 5)
3. Health (beat 3)
4. Architecture (beat 6)
5. Results / COOL / weaknesses (beats 7–9) — offline docs
6. Team and hook (beats 1–2) — webcam only
7. Close (beat 10)

Keep the raw takes. Edit them into timeline order afterwards.

---

## 0:00–0:08 — Team (screen: webcam)

> "I'm Neem Jangbu Lama. This is ForgeSight — a visual inspector for rolled steel."

Eight seconds, face and name. The submission requires the video to show the team.

## 0:08–0:28 — Hook (screen: one NEU photograph, full frame)

> "A photograph goes in. The system decides what should happen next — and
> sometimes acts on the physical world and looks again. That second part is the
> hard part, and it's what this is about."

## 0:28–0:48 — It is live, and it is OpenCV 5 (screen: terminal)

```powershell
curl.exe http://<IP>/health
```

> "This is the demo API, running right now on an AWS Graviton4 instance.
> `/health` returns OpenCV 5.0.0 — the version the competition requires, verified
> at request time rather than claimed in a README. The COOL measurement is a
> separate, library-only benchmark; I'll come back to it."

Pause on `"opencv": "5.0.0"`. Do **not** say the API is running COOL: it is not,
and `docs/DEPLOYMENT.md` says so in its first section.

## 0:48–1:52 — The loop (screen: browser, the main event)

1. Open `http://<IP>`. Bookmarks bar off (Ctrl+Shift+B).
2. Click the **scratches** sample button.
3. Click **soft focus**. The capture visibly blurs and the decision card says the
   settings were degraded.
4. Click **Run closed loop**. It takes a few seconds.
5. Expected result: decision **human review**, `iterations` **2**, a work order
   id, and under `final capture settings` the value `"focus_blur": 0.0`. The
   capture image is replaced by the **re-rendered frame** and visibly sharpens.
   The trace shows `applied 'steady_and_refocus' -> focus_blur 4.0 -> 0.0`.

Narrate over it, do not read the JSON:

> "The quality gate refuses to classify this frame. It doesn't just say no — it
> names the physical fix: *steady the camera and refocus*. So it changes a camera
> setting and re-renders the scene. This is not a patched image. The next
> perception is of a genuinely different frame, and the agent escalates with a
> work order."

6. Click **Approve**. The audit line reads
   `closed by human at <timestamp> · audit 2 events (closed)`.

> "Approval is a person, recorded with a timestamp. The model cannot close its
> own work order."

## 1:52–2:17 — The independent veto (screen: browser, `/inspect`)

There is **no clean sample in the demo set**, so this beat is a defect frame that
the classifier is fooled into calling clean:

1. Click the **rolled-in_scale** sample.
2. Click **specular glare**.
3. Click **Analyse** (not the loop).

Expected result: `predicted class no_defect`, `confidence 0.915`, `anomaly
regions 2`, decision **human review**, and the why line:
`classifier and localiser disagree - do not auto-pass`.

> "The classifier is 91% sure this surface is clean. The localiser finds two
> regions of structure. Two independent perceptions disagree, so the agent
> refuses to auto-pass. A weak second opinion is still useful as a veto."

Backup: **scratches** + **sensor noise** also produces this veto, but do not use
that preset — see the note at the bottom.

## 2:17–2:37 — Architecture (screen: `docs/architecture.png`, full screen)

Open the PNG in Photos, full screen. Pan once, slowly.

> "OpenCV 5 does all the perception — quality metrics, a Gabor descriptor, and a
> multi-scale localiser. The agent decides from that evidence. The API runs on a
> Graviton instance in AWS; the benchmark runs on the same AMI."

No more than 20 seconds. This is a required element, not a talking point.

## 2:37–3:32 — What it measures (screen: `docs/EVALUATION.md` in an editor)

Use Ctrl+G to jump, not scrolling: **line 116** (classification) then **line 147**
(triage). Read four numbers and only four:

> "1800 unique source photographs — NEU-CLS and NEU-DET turned out to be the same
> survey, so both are deduplicated and split by source image.
> **92.75% accuracy on seven classes**, held out.
> **Zero missed defects out of 540**, and auto-pass precision exactly **1.000** —
> nothing we released automatically was defective.
> And confidence is calibrated: **ECE 0.0361**, temperature fitted on validation."

If asked why not report sensitivity 1.0: the rule-of-three lower bound,
≥ 99.44% at 95%, is in the doc. Say it only if asked.

## 3:32–4:22 — The COOL result (screen: `docs/BENCHMARK-RESULTS.md`, Ctrl+G → line 9)

> "We ran the 3-arm benchmark on Graviton4. This is a **library-only** A/B — the
> benchmark swaps the OpenCV build under the same feature code; the demo API you
> just saw runs pip OpenCV 5.0.0, so I'm not claiming COOL sits in the request
> path. **COOL is 6% slower than stock OpenCV 5 on our workload** — 0.94x. COOL
> accelerates Canny by 1.57x, but Canny isn't in our hot path; our pipeline is
> dominated by a Gabor `filter2D`, and COOL leaves it at exactly 1.00x. We could
> have rewritten our feature bank to flatter COOL. We didn't, because that would
> have been optimising for the award rather than the product."

Then the retraction, briefly:

> "Three claims we'd made earlier don't survive re-measurement, and we've
> withdrawn all three: the COOL-beats-stock claim was the hardware, not COOL;
> our `c8g.medium` price was wrong by 1.8x; and a 2.20x speedup was really 1.83x."

## 4:22–4:52 — What is still weak (screen: `docs/EVALUATION.md`)

Ctrl+G → **line 362** ("Localisation — rebuilt, still the weak part"), then
**line 570** ("Failure cases and limitations"). Note the file has two sections
numbered 4 and 5; go by the heading text.

> "Localisation is the weak part: macro F1 0.2247. We found that 62% of the boxes
> we missed were the right defect at the wrong size — not defects we failed to
> see — so we raised our proposal ceiling from 0.35 to 0.55. Macro F1 moved
> 0.2200 to 0.2247. We could have quoted the 58% ceiling gain. We didn't, because
> the score is what matters and the score barely moved.
>
> The old 0.19 was tuned on the images it was reported on. We found that and
> withdrew it. `no_defect` rests on 269 clean crops, and there's no LLM planner —
> it's a measured gate, because we can defend that."

Never end on your best number.

## 4:52–5:00 — Close (screen: the running instance)

> "ForgeSight. Live on AWS, OpenCV 5, public data only. Report, benchmark and
> evaluation are in the repo — including everything we got wrong."

## Shot checklist

- [ ] `python eval\smoke_test.py` is 26/26 **before** recording
- [ ] prompt changed to `PS>`; `cls` so no username path is visible
- [ ] endpoint `up`, URL loads, `/health` shows 5.0.0
- [ ] `soft focus` visibly blurs; loop ends `focus_blur: 0.0` and sharpens
- [ ] loop returns a work order; approve prints `audit 2 events`
- [ ] `rolled-in_scale` + `specular glare` shows `no_defect` @ 0.915, 2 regions,
      **human review**, with the why line
- [ ] architecture PNG shown for at least 15 seconds
- [ ] **sensor noise is never pressed on camera**
- [ ] `/docs` left closed — no tab with credentials
- [ ] instance terminated afterwards: `python aws\deploy_endpoint.py down`

## Things not to say

- Don't call it "AI-powered" or "ML-driven" — it's OpenCV features plus a NumPy
  logistic regression, and claiming otherwise invites a question you can't answer.
- Don't claim the loop is "self-correcting in production". It's a capture-settings
  **simulator**, and the UI labels it as such. Say "simulator mode" out loud.
- Don't oversell the COOL result. 0.94x slower, on our workload, stated plainly.
- Don't describe the API as running COOL. It runs pip OpenCV 5.

## Known issue, not in this video

The UI's **sensor noise** preset can auto-pass a defective sample: the preset adds
Gaussian sigma 18, but the quality gate estimates 9.64–10.75 against a 10.8 limit,
so the gate does not fire (and Laplacian sharpness reads ~2900 because noise looks
like detail). In a 6-sample probe, 3 defective samples were passed. The ablation's
"false accepts 0/200" counts *clean* images that were auto-passed, not defects
that were passed, so it does not cover this. It is being fixed before submission;
until then, no shot uses that preset.
