# Release audit outputs

Generate `release_audit.json` and `checksums.json` from the final clean release
tree immediately before tagging:

```powershell
root-teller audit-release --release-root . --output-dir docs/audit
```

Generated outputs describe the exact files present at generation time. Do not
reuse an audit report after renaming, adding, or removing release files.

