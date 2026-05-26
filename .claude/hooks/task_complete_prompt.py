#!/usr/bin/env python3

import json

message = """
Before considering this task complete, perform a self-review.

Check the following carefully:

1. Did you add or update tests where appropriate?
2. Did you run the relevant validation/test commands?
3. Did you update API documentation if APIs changed?
4. Did you update CLAUDE.md if architecture or workflows changed?
5. Are there edge cases or regressions?
6. Is the implementation production ready?
7. Are there any TODOs, hacks, or temporary fixes remaining?
8. Review all changed files critically.

If anything is incomplete, continue working before finishing the task.
"""

print(json.dumps({
    "decision": "block",
    "reason": message
}))
