# Security policy

This repository is research/reference software and is not approved for clinical deployment. Public issues must contain only synthetic or clearly de-identified material.

Do not publicly report suspected PHI or credential exposure. Report it privately to the maintainers with a minimal synthetic reproduction.

## Non-negotiable deployment controls

- Keep FHIR writes disabled until an organization-specific review approves an authenticated, auditable persistence path.
- Do not enable external telemetry for PHI-bearing deployments.
- Do not log document bodies or evidence text in centralized logs.
- Treat all supplied document text as untrusted input.
- Use least-privilege, read-only FHIR credentials wherever possible.
