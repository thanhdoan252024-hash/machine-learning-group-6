# 11 — Run Order Quick Guide

## Quick core audit
`EXECUTION_PROFILE = "quick"` → P00–P08 core validation only.

## Screening
`EXECUTION_PROFILE = "screening"` → additionally run P09–P10.

## Full study
`EXECUTION_PROFILE = "full"` → P09–P12 including multi-seed and 5620-sample final run.

Never jump to a later phase with variables left over from an older runtime.
