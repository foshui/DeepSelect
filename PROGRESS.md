# DeepSelect build diagnosis

**Status**: 🔄 In Progress
**Started**: 2026-10-31 UTC
**Last updated**: 2026-10-31 10:23 UTC
**Current stage**: 1. Diagnose and fix build

## TODO
- [ ] 1. Diagnose and fix build
- [ ] 2. Verify extension and comparison

---

## Detailed Log

User reports Ascend extension compilation failure after missing backend import.

### [10:20] 1. Diagnose and fix build
Starting...

### [10:23] Build reproduction findings
Fixed hardcoded aarch64 CANN paths by selecting the host architecture. Rebuild still fails: CANN 9.1.0 lacks asc_pack_to_low, asc_squeeze_and_storeunalign variants, and asc_duplicate_highest; bisheng also reports duplicate pair assignment declarations in GCC 13 headers. CANN 9.2.0 is referenced by setup.py, but a build with that version has not been verified. No NPU device is exposed in this tool environment. User is considering CANN 9.2.0; no installation performed.
