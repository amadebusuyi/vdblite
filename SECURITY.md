# Security policy

## Supported versions

Security fixes go into the latest release. While VDBLite is 0.x, older minor
versions are not patched.

## Reporting a vulnerability

Please don't open a public issue. Report it privately through GitHub's
**Security → Report a vulnerability** on this repository. Include what you
found, how to reproduce it (a small `.vdb` file or program helps), and what
an attacker could do with it.

You'll get a reply within a week. Once a fix is released, the advisory is
published with credit to you unless you'd rather stay anonymous.

Of particular interest: memory unsafety or panics when opening a corrupted or
hostile `.vdb` file, and anything that loses or corrupts committed data.
