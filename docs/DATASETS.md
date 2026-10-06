# Third-party datasets are NOT committed. Obtain them yourself.

These are large, publicly available research datasets. They are excluded from
the repository (and `.gitignore`d) so this project does not redistribute someone
else's data, and so a clone stays small. Every number in `docs/EVALUATION.md` is
reproducible from them.

## Required

### NEU-CLS — 1800 images, 6 surface-defect classes
Classification subset of the NEU (Northeastern University) rolled-steel surface
defect dataset. 200x200 grayscale BMP, 300 images per class.

Classes: `Cr` crazing, `In` inclusion, `Pa` patches, `PS` pitted surface,
`RS` rolled-in scale, `Sc` scratches.

### NEU-DET — 1800 images + 1800 VOC-style XML box sets
Detection subset of the same dataset: JPEG images plus `ANNOTATIONS/*.xml` with
`<bndbox>` per defect.

Used for two things:
1. the second half of the classifier's training set (inference runs on these
   captures, and a model trained only on the CLS BMPs was miscalibrated on them);
2. background crops **outside** every annotation box, which supply the
   `no_defect` class the dataset itself lacks.

### Layout expected by the code

```
<project root>/
  NEU-CLS/NEU-CLS/*.bmp
  NEU-DET/NEU-DET/IMAGES/*.jpg
  NEU-DET/NEU-DET/ANNOTATIONS/*.xml
```

Paths are set in `eval/train_classifier.py` (`CLS_DIR`, `DET_ROOT`).

## Optional, not used

`archive/DeepPCB` (PCB defects) and `archive (1)/elpv-dataset-master` (solar
panel EL) were downloaded while choosing a domain. **They are unused.** The
project deliberately covers one domain; keeping four would mean claiming
evidence we do not have.

## Attribution

NEU surface defect data originates from the Northeastern University (China)
rolled-steel surface defect dataset, distributed for academic research. Cite the
original dataset when publishing results. Check the current licence terms before
any commercial use.

## Small derived samples

`samples/neu/*.jpg` are six 200x200 NEU photographs, re-encoded at JPEG quality
70, shipped so the AWS benchmark measures real image analysis on every arm
without needing the full dataset on the instance. They remain NEU data.
