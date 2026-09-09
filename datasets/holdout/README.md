# Real-video holdout

Place real clips here (`*.mp4`) together with a `<clip>.labels.json` produced by
`failsafe corpus label <clip>`. This corpus is NEVER used by the search / discovery loop; it is only
used as a reality check that a discovered policy does not merely exploit the synthetic generator.
