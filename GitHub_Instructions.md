# GitHub Workflow for Working Across Two Laptops

## Objective

Maintain the same project on two laptops by using GitHub as the central source of truth.

**Example Setup**

- **Laptop A** → Office Laptop
- **Laptop B** → Personal Laptop
- **GitHub Repository** → Central Repository

---

# One-Time Setup

## 1. Install Git

Download Git from:

https://git-scm.com/

Verify installation:

```bash
git --version
```

---

## 2. Clone the Repository

Using HTTPS

```bash
git clone https://github.com/<username>/<repository>.git
```

Using SSH (Recommended)

```bash
git clone git@github.com:<username>/<repository>.git
```

Do this on **both laptops**.

---

## 3. Configure Git (First Time Only)

```bash
git config --global user.name "Your Name"

git config --global user.email "your@email.com"
```

Verify

```bash
git config --list
```

---

# Daily Workflow

Always follow the same sequence.

```
Pull
   ↓
Modify Code
   ↓
Commit
   ↓
Push
```

This prevents almost all synchronization problems.

---

# Step 1 – Pull Latest Changes

Before writing any code:

```bash
git pull origin main
```

If your default branch is **master**

```bash
git pull origin master
```

This downloads the latest version from GitHub.

---

# Step 2 – Modify Your Code

Edit your project normally.

Examples:

- Python files
- Streamlit apps
- Jupyter notebooks
- Documentation
- Markdown files

---

# Step 3 – Check What Changed

```bash
git status
```

Example output

```
modified: app.py

modified: README.md

modified: requirements.txt
```

---

# Step 4 – Stage Files

Stage everything

```bash
git add .
```

Or stage individual files

```bash
git add app.py
```

---

# Step 5 – Commit Changes

Create a meaningful commit message.

```bash
git commit -m "Added OCR module"
```

Good examples

```
Fixed login bug

Updated README

Improved RAG pipeline

Added Power BI dashboard

Optimized SQL query

Refactored preprocessing module
```

---

# Step 6 – Push to GitHub

```bash
git push origin main
```

Your latest changes are now stored on GitHub.

---

# Working on the Second Laptop

When switching laptops:

```bash
git pull origin main
```

Everything becomes synchronized.

Continue working as usual.

After finishing:

```bash
git add .

git commit -m "Implemented feature X"

git push origin main
```

---

# Complete Example

## Laptop A

```bash
git pull origin main

# Modify files

git status

git add .

git commit -m "Added OCR pipeline"

git push origin main
```

---

## Laptop B

Next day

```bash
git pull origin main
```

You now have:

- Latest Python files
- Latest notebooks
- Latest documentation
- Latest commits

---

# Merge Conflicts

A merge conflict occurs when:

- Laptop A edits a file
- Laptop B edits the same file
- Neither laptop pulled the other's changes first

Example

```python
<<<<<<< HEAD
print("Laptop B")
=======
print("Laptop A")
>>>>>>> main
```

To resolve:

1. Open the file.
2. Decide which code to keep.
3. Remove the conflict markers.
4. Save the file.

Then run

```bash
git add app.py

git commit -m "Resolved merge conflict"

git push origin main
```

---

# Golden Rule

Always

```
Pull

↓

Modify

↓

Commit

↓

Push
```

Never

```
Modify on Laptop A

↓

Modify same file on Laptop B

↓

Push both
```

This usually causes conflicts.

---

# Useful Git Commands

### Check Status

```bash
git status
```

---

### View Commit History

```bash
git log --oneline
```

---

### Pull Latest Changes

```bash
git pull origin main
```

---

### Push Changes

```bash
git push origin main
```

---

### Stage All Files

```bash
git add .
```

---

### Commit

```bash
git commit -m "Your message"
```

---

### View Differences

```bash
git diff
```

---

### Check Current Branch

```bash
git branch
```

---

### Switch Branch

```bash
git checkout branch-name
```

---

### Create New Branch

```bash
git checkout -b feature/new-feature
```

---

# Recommended .gitignore for Python Projects

```gitignore
# Bytecode
__pycache__/
*.pyc

# Virtual Environments
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# VS Code
.vscode/

# Environment Variables
.env

# Logs
*.log

# OS Files
.DS_Store
Thumbs.db

# Python Build
build/
dist/

# Large Data
data/

# Models
models/
checkpoints/

# Temporary Files
*.tmp

# Cache
.cache/
```

---

# GitHub Desktop Workflow (GUI)

If you prefer a graphical interface:

1. Open GitHub Desktop.
2. Click **Fetch Origin**.
3. Click **Pull Origin**.
4. Make changes.
5. Enter a commit message.
6. Click **Commit to Main**.
7. Click **Push Origin**.

Repeat the same process on the second laptop.

---

# Pull Request vs Pull

Many beginners confuse these two terms.

## git pull

Downloads the latest changes from GitHub to your local machine.

Example

```bash
git pull origin main
```

Use this every day.

---

## Pull Request (PR)

A Pull Request is used when:

- Working with a team.
- Developing features on separate branches.
- Requesting code review before merging into the main branch.

If you're the only developer working directly on the `main` branch, you generally do **not** need Pull Requests for syncing your laptops.

---

# Best Practices

- Pull before starting work.
- Commit frequently with clear messages.
- Push at the end of every work session.
- Avoid editing the same file on two laptops simultaneously.
- Keep commits small and focused.
- Use `.gitignore` to exclude virtual environments, secrets, logs, datasets, and model artifacts.
- Create feature branches for major changes.
- Never commit passwords, API keys, or `.env` files.

---

# Recommended Daily Routine

### Morning

```bash
git pull origin main
```

Work on the project.

---

### Before Lunch

```bash
git add .

git commit -m "Completed dashboard improvements"

git push origin main
```

---

### Evening

```bash
git pull origin main

git add .

git commit -m "Completed API integration"

git push origin main
```

---

### On the Second Laptop

```bash
git pull origin main
```

Start coding immediately with the latest project.

---

# Summary Workflow

```text
Laptop A
   │
   │ git push
   ▼
+------------------+
|      GitHub      |
+------------------+
   ▲
   │ git pull
   │
Laptop B
```

GitHub acts as the **single source of truth**, ensuring both laptops stay synchronized when you consistently follow the **Pull → Modify → Commit → Push** workflow.