# ForgeSight — 5-minute demo video script

Target 4:45–5:00. Record at 1920x1080, 30 fps, voiceover over screen capture.
**Everything below is already implemented and already verified** — this is a
running order for what exists, not a wish list.

The single most important decision: **lead with the closed loop, not the
metrics.** Agentic Vision is judged on whether visual evidence changes a later
action. The metrics section is proof, not the hook.

---

## Before you record

```powershell
aws login                                    # session expires; verify first
python aws\deploy_endpoint.py up --ttl-hours 3
python aws\deploy_endpoint.py status          # copy the URL
aws sts get-caller-identity                  # must succeed
```

Keep the terminal visible in a second window for the `/health` shot. Kill the
instance when done — see the cost table in `docs/DEPLOYMENT.md`.

---

## 0:00–0:20 — Hook (screen: title card over one steel surface photo)

> "A photograph goes in. The system decides what should happen next — and
> sometimes acts on the physical world and looks again. That second part is the
> hard part, and it's what this is about."

On screen: one NEU photograph, full frame. No UI clutter yet.

---

## 0:20–0:40 — It is live, and it is OpenCV 5 (screen: terminal)

```powershell
curl http://<URL>/health
```

Narration: *"This is running right now on a COOL Graviton4 instance in AWS.
`/health` returns OpenCV 5.0.0 — the version the competition requires, verified
at request time rather than claimed in a README."*

Pause on the `"opencv": "5.0.0"` field. This is a 20-second credibility beat.

---

## 0:40–1:50 — The loop (screen: browser, the main event)

**Use the `soft_focus` preset.** From `docs/EVALUATION.md` §4d it is the only
condition where the gate refuses, names a fix, applies it, *and* converges in
one step. Every other condition is a weaker demonstration.

1. Open `http://<URL>`.
2. Click a **scratches** sample.
3. Click the **soft focus** preset button — the capture visibly blurs.
4. Hit the loop button. Narrate over it, do not read the JSON:
   > "The quality gate refuses to classify this frame. It doesn't just say no —
   > it names the physical fix: *steady the camera and refocus*. So it changes a
   > camera setting and **re-renders the scene**. This is not a patched image."
5. The trace appears. Narrate:
   > "The next perception is of a genuinely different frame. The agent now
   > classifies, and escalates with a region and a work order."
6. Open the work order, click approve once, show the audit line.

**Fallback if the demo misbehaves on camera:** this exact trace is captured
verbatim in `docs/DEPLOYMENT.md` lines 63–78 and runs on the deployed instance.
Paste it as an overlay rather than re-rolling ten times.

---

## 1:50–2:20 — The independent veto (screen: `/inspect` on a clean sample)

> "Classification and localisation are two separate perceptions. When the
> classifier says clean but the localiser finds structure, the agent refuses to
> auto-pass. A weak second opinion is still useful as a veto."

One clean sample through `/inspect`. Keep it short — 20 seconds, then move.

---

## 2:20–3:20 — What it measures (screen: `docs/EVALUATION.md` §2–3, scroll slowly)

Read these four and only these four. Do not read the table.

> "1800 unique source photographs — NEU-CLS and NEU-DET turned out to be the
> same survey, so both are deduplicated and split by source image.
> **92.75% accuracy on seven classes**, held out.
> **Zero missed defects out of 540**, and auto-pass precision exactly **1.000** —
> nothing we released automatically was defective.
> And confidence is calibrated: **ECE 0.034**, temperature fitted on validation."

If a judge asks why not just report sensitivity 1.0: the rule-of-three lower
bound, ≥ 99.44% at 95%, is in the doc. Mention it only if asked.

---

## 3:20–4:15 — The COOL result (screen: `docs/BENCHMARK-RESULTS.md` §1)

This is the most memorable 55 seconds in the video, because it is a negative
result you volunteered.

> "We ran the 3-arm benchmark on Graviton4. **COOL is 6% slower than stock OpenCV
> 5 on our workload** — 0.94x. COOL accelerates Canny by 1.57x, but Canny isn't
> in our hot path; our pipeline is dominated by a Gabor `filter2D`, and COOL
> leaves it at exactly 1.00x. We could have rewritten our feature bank to flatter
> COOL. We didn't, because that would have been optimising for the award rather
> than the product."

Then, briefly, the retraction — this is what separates an honest submission:

> "Three claims we'd made earlier don't survive re-measurement, and we've
> withdrawn all three: the COOL-beats-stock claim was the hardware, not COOL;
> our `c8g.medium` price was wrong by 1.8x; and a 2.20x speedup was really 1.83x."

---

## 4:15–4:50 — What is still weak (screen: `docs/EVALUATION.md` §5)

Never end on your best number. State the real limit.

> "Localisation is the weak part and we won't pretend otherwise: macro F1 0.219.
> The honest reason is a ceiling — our proposal generator's own oracle recall is
> 0.346, so no detector built on it can do better. The previous number, 0.19, was
> tuned on the same images it was reported on. We found that, fixed it, and
> withdrew the claim.
>
> `no_defect` rests on only 269 clean crops. Glare detection doesn't work and is
> disabled. There's no LLM planner here — it's a transparent, measured gate,
> because we can defend that and we can't defend a reasoning chain."

---

## 4:50–5:00 — Close (screen: the running instance)

> "ForgeSight. Live on AWS, OpenCV 5, public data only. Report, benchmark and
> evaluation are in the repo — including everything we got wrong."

---

## Shot checklist

- [ ] `aws sts get-caller-identity` succeeds **before** recording
- [ ] endpoint `up`, URL loads, `/health` shows 5.0.0
- [ ] `soft_focus` preset visibly blurs on click
- [ ] loop converges and shows a work order ID
- [ ] approve action produces an audit line
- [ ] no file paths with your username visible in the terminal
- [ ] `/docs` left closed — no accidental tab with credentials
- [ ] instance terminated afterwards

## Things not to say

- Don't call it "AI-powered" or "ML-driven" — it's OpenCV features plus a NumPy
  logistic regression, and claiming otherwise invites a question you can't answer.
- Don't claim the loop is "self-correcting in production". It's a capture-settings
  **simulator**, and the UI labels it as such. Say "simulator mode" out loud.
- Don't oversell the COOL result. 0.94x slower, on our workload, stated plainly.