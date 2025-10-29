# Git Setup Instructions

## Setting Up Correct Author Information

This repository should only show contributions from `abhiramavarma`.

### Option 1: Fresh Start (Recommended for New Repository)

If you're starting fresh or don't mind losing commit history:

```bash
# Remove old remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/abhiramavarma/Lip2Text.-.git

# Ensure local config is set
git config user.name "abhiramavarma"
git config user.email "abhiramavarma@gmail.com"

# Stage all files
git add .

# Create a fresh commit with correct author
git commit -m "Initial commit: Lip2Text project setup"

# Push to new repository
git push -u origin main --force
```

### Option 2: Rewrite History (Preserves Commit History)

If you want to keep commit history but fix authorship:

```bash
# Ensure local config is set
git config user.name "abhiramavarma"
git config user.email "abhiramavarma@gmail.com"

# Run the author fix script
./fix_git_authors.sh

# Verify the changes
git log --format='%an %ae' | sort -u

# Force push to update remote
git push --force --all
```

**Note:** Force pushing rewrites remote history. Only do this if you're sure, or if the repository is new.

### Verify Setup

Check that all commits now show your name:
```bash
git log --format='%an %ae' | sort -u
```

You should only see:
```
abhiramavarma abhiramavarma@gmail.com耗
```

