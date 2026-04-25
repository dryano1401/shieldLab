# Git Repository Setup Instructions

Follow these steps to create your GitHub repository for ShieldLab.

## Step 1: Initialize Git Repository Locally

Open a terminal/command prompt in your ShieldLab directory and run:

```bash
cd /path/to/shieldlab  # Navigate to your project directory
git init
git add .
git commit -m "Initial commit: ShieldLab Monte Carlo shielding toolchain"
```

## Step 2: Create Repository on GitHub

1. Go to [github.com](https://github.com) and log in
2. Click the **"+"** icon in the top-right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `shieldlab` (or your preferred name)
   - **Description**: "Monte Carlo radiation shielding simulation toolchain for nuclear medicine and diagnostic radiology"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 3: Connect Local Repository to GitHub

GitHub will show you commands. Use these (replace YOUR_USERNAME with your GitHub username):

```bash
git remote add origin https://github.com/YOUR_USERNAME/shieldlab.git
git branch -M main
git push -u origin main
```

**Alternative with SSH** (if you have SSH keys set up):
```bash
git remote add origin git@github.com:YOUR_USERNAME/shieldlab.git
git branch -M main
git push -u origin main
```

## Step 4: Verify Upload

1. Refresh your GitHub repository page
2. You should see all files uploaded
3. The README.md will be displayed on the main page

## Step 5: Update Repository Links

After creating the repository, update these files with your actual GitHub URL:

1. **README.md**: Update the contact/issues section
2. **setup.py**: Line 13, change `YOUR_USERNAME` to your GitHub username
3. **CITATION.cff**: Line 9 and 10, update the repository URL

Then commit and push changes:
```bash
git add README.md setup.py CITATION.cff
git commit -m "Update repository URLs"
git push
```

## Step 6: Add Topics (Optional but Recommended)

On your GitHub repository page:
1. Click the **gear icon** next to "About"
2. Add topics like:
   - `monte-carlo`
   - `radiation-shielding`
   - `medical-physics`
   - `nuclear-medicine`
   - `gate-simulation`
   - `python`
3. Save changes

## Step 7: Enable GitHub Actions (Optional)

The workflow file is already included (`.github/workflows/test.yml`). It will automatically:
- Run tests on push/pull requests
- Test on multiple OS and Python versions
- Verify basic functionality

GitHub Actions should activate automatically on your next push.

## Common Git Commands for Future Updates

### Making Changes
```bash
# See what files changed
git status

# Add specific files
git add shieldLabSim.py

# Or add all changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

### Creating Branches
```bash
# Create and switch to new branch
git checkout -b feature-name

# Push branch to GitHub
git push -u origin feature-name
```

### Pulling Latest Changes
```bash
# Get latest changes from GitHub
git pull
```

## Troubleshooting

### "Permission denied (publickey)"
You need to set up SSH keys or use HTTPS with personal access token.

**Quick fix**: Use HTTPS URL instead:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/shieldlab.git
```

### "Repository not found"
Make sure you:
1. Created the repository on GitHub
2. Used the correct username in the URL
3. Have access to the repository (check if it's under your account)

### Large files warning
Git has a file size limit (~100MB). If you have large simulation output files:

1. Make sure they're listed in `.gitignore`
2. Only commit code, not output data
3. For large files you need to track, use Git LFS:
   ```bash
   git lfs install
   git lfs track "*.raw"
   ```

## Next Steps

1. **Add collaborators** (if working with others):
   - Go to repository Settings → Collaborators
   - Invite by GitHub username or email

2. **Set up branch protection** (for production code):
   - Settings → Branches → Add rule
   - Require pull request reviews before merging

3. **Create releases** (when ready):
   - Go to Releases → Create a new release
   - Tag version (e.g., `v0.1.0`)
   - Add release notes

4. **Add badges to README** (optional):
   - ![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
   - ![License](https://img.shields.io/badge/license-MIT-green)
   - ![Tests](https://github.com/YOUR_USERNAME/shieldlab/workflows/ShieldLab%20Tests/badge.svg)

## Resources

- [GitHub Docs](https://docs.github.com)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Pro Git Book](https://git-scm.com/book/en/v2) (free online)

Good luck with your repository! 🚀
