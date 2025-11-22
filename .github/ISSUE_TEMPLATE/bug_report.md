---
name: Bug report
about: Report a problem or unexpected behavior when using dmeth as a library
title: ''
labels: bug
assignees: ''

---

**Description**

Provide a clear and concise description of the issue or unexpected behavior observed when using **dmeth** in Python.

---

**Code Example to Reproduce**

Please include a minimal reproducible snippet:

```python
import numpy as np
import pandas as pd

from dmeth.core.analysis.core_analysis import fit_differential
from dmeth.core.analysis.validation import validate_design, validate_contrast

# Setup
M = pd.DataFrame(
    [[0.1, 0.2], [0.3, 0.4]],
    index=["cg1", "cg2"],
    columns=["s1", "s2"]
)

pheno = pd.DataFrame({"group": ["case", "control"]}, index=["s1", "s2"])

design = validate_design(pheno["group"])

contrast = validate_contrast(design, "case-control")

# Function call
results = fit_differential(
    M=M,
    design=pd.DataFrame(design, index=M.columns),
    contrast=contrast,
    shrink="smyth",
    robust=True,
)

print(results.head())
# output
# output
# output
```

Include any necessary input data structure or shapes to reproduce the issue.

---

**Expected Behavior**

Describe what you expected the function to return or do (e.g., return a DataFrame with columns \[`logFC`, `pval`, …]).

---

**Actual Behavior**

Describe what actually happened (e.g., exception message, NaNs in results, mismatched output shape).

Paste full error traceback here if available.

---

**Environment**

Please provide:
- **OS**: \[e.g. macOS 14, Ubuntu 22.04, Windows 11]
- **EDI/Console**: \[e.g. VS Code]
- **Python version**: \[e.g. 3.11]
- **dmeth version**: \[e.g. 0.2.0]
- **Installation method**: \[e.g. `pip install .`, `pip install -e .`]

---

**Additional Context**

Add any other context that might help diagnose the issue:
- Dataset type (e.g. 450k, EPIC)
- Function(s) involved (`fit_differential`, `_add_group_means`, etc.)
- Whether it happens consistently or only under certain data conditions.