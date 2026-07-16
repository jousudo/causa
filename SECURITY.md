# Security Policy

## Supported versions

`causa` is pre-`v0.1.0` — no version has been tagged yet, so none is under a support
commitment. Once `v0.1.0` ships, this section will list which lines receive security fixes.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/jousudo/causa/security/advisories/new) for this
repository, rather than opening a public issue.

`causa` is a standard-library-only, CGO-free Go library with no network I/O, no file I/O beyond
what a caller explicitly does with the data it returns, and no third-party dependencies — the
realistic attack surface is small (e.g. a panic or resource exhaustion on adversarial input to
an exported function). Still, if you find one, please report it privately so a fix can ship
before the details are public.

There is no bug bounty. A best-effort acknowledgment and timeline will be provided once a
report is triaged.
