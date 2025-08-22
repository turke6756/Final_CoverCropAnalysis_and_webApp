# Git Workflow Best Practices for Claude Code

## 1. Verify Git Repository Status

**Never trust environment metadata alone** - always verify with actual commands:

```bash
# Check if directory is a Git repository
git status

# Find .git directory if status fails
find . -name ".git" -type d

# Check remote repository URL
git remote -v
```

## 2. GitHub CLI Setup (One-Time Only)

**Note: GitHub CLI setup is typically done once per system**

If not already installed:
```bash
# Install (Ubuntu/WSL)
sudo apt update && sudo apt install gh

# Authenticate (one-time setup)
gh auth login
# Choose: GitHub.com → HTTPS → Yes (authenticate Git)
```

**Always verify authentication before starting work:**
```bash
gh auth status
```

## 3. Pull Request Handling Workflow

**RECOMMENDED: Review → Implement Locally → Test → Push**

### Step 1: Authenticate & Check PRs
```bash
# Verify authentication
gh auth status

# List all open PRs with details
gh pr list
```

**Present findings to user:**
- Report what PRs were found (number, title, branch, date)
- Ask user which PR to implement
- Example: "Found PR #9: 'Use precomputed streak metrics' - should I implement this?"

### Step 2: Review Selected PR
```bash
# View specific PR details
gh pr view [PR_NUMBER]

# Get PR diff to understand changes
gh pr diff [PR_NUMBER]
```

**Communicate to user:**
- Summarize what changes the PR makes
- Explain the scope and potential impact
- Get user confirmation before proceeding

### Step 3: Implement Changes Locally
- Review the PR changes manually
- Implement the changes in local files
- DO NOT merge the PR online yet
- **Status update**: "Implementing PR #X changes locally..."

### Step 4: Test Thoroughly
- Run all tests locally
- Verify functionality works as expected
- Check for any breaking changes
- **Status update**: "Testing implemented changes..."

### Step 5: Deploy if Tests Pass
```bash
# If tests pass - push local changes
git add .
git commit -m "Implement PR #X: [description]"
git push

# Close the PR (since changes are now in main)
gh pr close [PR_NUMBER]
```

**Status update**: "Tests passed! Changes pushed to main and PR #X closed."

### Step 6: Discard if Tests Fail
```bash
# If tests fail - discard local changes
git restore .
# Online repo remains stable, no cleanup needed
```

**Status update**: "Tests failed. Local changes discarded. Online repo remains stable."

## 4. Key Principles

- **Local First**: Always test locally before affecting online repository
- **Stable Main**: Keep online main branch stable until changes are verified
- **Clean History**: Avoid merge-revert cycles by testing first
- **Verify Commands**: Use `git status` and actual Git commands over environment metadata

## 5. Common Commands

```bash
# Repository status
git status
gh repo view

# Branch management
git branch
git checkout -b [branch-name]

# PR management
gh pr list
gh pr view [number]
gh pr close [number]
```