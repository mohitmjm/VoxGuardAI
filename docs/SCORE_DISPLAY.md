# Website score interpretation

The light-only frontend now uses user-selected presentation bands:

| Raw model score | Display label |
| --- | --- |
| 0 <= score < 0.10 | Strong human-like signal |
| 0.10 <= score < 0.50 | Mostly human-like signal |
| 0.50 <= score < 0.80 | AI-like signal |
| 0.80 <= score <= 1 | Strong AI-like signal |

The displayed percentage is 100 times the uncalibrated score, not a probability of AI generation, confidence, or fraction of manipulated speech. Band selection uses the raw score, not its rounded display value. Cutoffs are user-chosen presentation rules and have not been fitted to representative validation data.

The backend response and default abstention policy are unchanged. The report's expandable details disclose the backend decision, and JSON exports retain the original classification while adding a separate display_interpretation object with validated=false. This distinction is intentional: presentation bands are not a validated model decision policy.

The landing page is shorter. The theme toggle, research dashboard, decorative waveform and repeated research sections are removed from the main view. Audio playback, upload validation, model readiness, segment seeking, processing details, cancellation, limitations and JSON export remain available.
