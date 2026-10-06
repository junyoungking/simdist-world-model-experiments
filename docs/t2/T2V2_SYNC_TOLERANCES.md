# T2-v2 synchronization tolerance

Frozen before outcomes: every mandatory effective-state field must have
maximum absolute env-0/env-1 difference at most `1e-5`; both environments
must be alive and the common-action history hash must match. This is the
strict equality threshold already used by the prior branch harness.
