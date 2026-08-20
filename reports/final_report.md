# Final report

Draft. Replace placeholders with measured numbers after training.

## Problem

Decide whether a paused user has finished speaking, especially on Indian English/Hindi audio, fillers, and short replies.

## Approach

Whisper Tiny encoder, attention pooling, compact MLP head, trained only on official Smart Turn v3.2 data. Threshold selected on a grouped validation split. Public test is unused until freeze.

## Results

- Official baseline:
- Head-only:
- Partial unfreeze:
- Indic slice (Hindi/Marathi):
- CPU INT8 size / p50 / p95:

## Failures

Document the top error categories here.

## Limitations

No extra Hinglish collection. No claim of production full-duplex quality.
