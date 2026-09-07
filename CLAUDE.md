# Claude instructions for Health ML Tracker

## After implementing new features or code changes

Always use the **test-validation** skill (`.claude/skills/test-validation/SKILL.md`) after completing any implementation task. This means running the local test suite and confirming all tests pass before telling the user the work is done or suggesting they push/commit.

Do not consider an implementation task complete until test-validation has been run and all tests pass.
