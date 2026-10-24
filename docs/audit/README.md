# Release audit outputs

Generate `release_audit.json` and `checksums.json` from the final clean release
tree immediately before tagging:

```powershell
root-teller audit-release --release-root . --output-dir docs/audit
```

Generated outputs describe the exact files present at generation time. Do not
reuse an audit report after renaming, adding, or removing release files.

The current audit distinguishes preserved historical configuration from the
revised reference contract. It checks packaging, credential/path exclusions,
the transcribed result table and documented method inventory. A passing audit
does not establish model/backbone parity, retrained fine-interval weights or
benchmark accuracy. Audit outputs omit their own files from the checksum scope.

