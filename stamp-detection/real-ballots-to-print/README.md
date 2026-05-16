# Real Ballots To Print

This folder contains a repeatable process for creating 50 synthetic ballot print masters. The images are intended to be printed on A3 paper, physically handled, scanned or photographed, and then labeled as real-world training data for the ballot stamp detector.

Generated files:

- `images/ballot_001.jpg` through `images/ballot_050.jpg`: individual high-resolution 300 DPI ballot images.
- `ballots_a3_print.pdf`: one ballot per A3 portrait page, centered with margins and preserving aspect ratio.
- `manifest.csv`: source image, stamp presence, intended placement category, appearance variant, and visible stamp box metadata.

The generated set contains 40 stamped ballots and 10 no-stamp hard negatives. Stamped ballots use the same stamp sizing and compositing approach as `src/tools/generate_synthetic.py`: stamp height is sampled as a small fraction of page height, the stamp is rotated without expanding its canvas, and the original stamp pixels are alpha-blended onto the ballot with random opacity. No-stamp ballots include handwriting-like vote numbers, specks, smudges, and marks near boxes without adding a stamp.

## Regenerate

From the repository root, run:

```powershell
& 'C:\Users\dhess\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' stamp-detection\real-ballots-to-print\generate_real_ballots.py
```

To get a different repeatable set, pass another seed:

```powershell
& 'C:\Users\dhess\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' stamp-detection\real-ballots-to-print\generate_real_ballots.py --seed 12345
```

The script only writes inside this folder. It refreshes `images/ballot_*.jpg`, `manifest.csv`, and `ballots_a3_print.pdf`.

## Print

Print `ballots_a3_print.pdf` on A3 portrait paper. Use actual size or 100% scale if your printer supports the margins. If the printer driver warns about clipping, use "fit to printable area"; the PDF already centers each ballot on the page and leaves margins.

## After Printing

1. Print the PDF on A3 paper.
2. Crumple, tear, fold, crease, mark, or lightly soil some ballots.
3. Scan or photograph the handled ballots under realistic lighting and focus conditions.
4. Put the scanned or photographed results in a separate dataset folder, not in this print-master folder.
5. Label the real scanned stamps in YOLO format.
6. Split the scanned data into train, validation, and test sets for detector training and evaluation.

Do not use these generated print masters as final real-world training images by themselves. Their purpose is to become physical paper samples first, then scanned or photographed real-world examples.
