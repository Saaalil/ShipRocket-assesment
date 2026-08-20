# Data card

## Sources

- Train: `pipecat-ai/smart-turn-data-v3.2-train` (~270,946 rows, ~41.4 GB)
- Test: `pipecat-ai/smart-turn-data-v3.2-test` (~31,527 rows, ~4.84 GB)

## Labels

- `endpoint_bool`: complete (`true`) vs incomplete (`false`)
- `midfiller`, `endfiller`: auxiliary metadata, not inference inputs
- `synthetic`: provenance
- `language`: used for slicing and optional Indic upsampling

## Split policy

Local 90/10 grouped split by `dataset` + id prefix, seed 42. Public test is unused for selection.

## Indic proxy

Hindi and Marathi rows plus English filler/short utterances from the official corpus. This is not claimed as true Hinglish unless transcripts show code-switching.

## Leakage risks

Speaker IDs are not always available. Residual same-speaker leakage is possible and must be reported.
