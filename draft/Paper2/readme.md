# Mechanistic Interpretability of Large Language Models

Scientific paper for the module **AI Frameworks & Tools**
Topic: **Mechanistic Interpretability with Sentiment Analysis as a Case Study**

---

# Repository Structure

```text
.
├── main.tex
├── cover.tex
├── glossary.tex
├── references.bib
├── logo.png
├── readme.md
├── .gitignore
├── chapters/
│   ├── 01_einleitung.tex
│   ├── 02_grundlagen.tex
│   ├── 03_modell_daten.tex
│   ├── 04_verhalten.tex
│   ├── 05_embedding_raum.tex
│   ├── 06_logit_lens.tex
│   ├── 07_attention.tex
│   ├── 08_activation_patching.tex
│   ├── 09_diskussion.tex
│   └── 10_fazit.tex
└── figures/
    └── (24 PNG-Abbildungen)
```

---

# Development Environment

This project uses:

* Windows 11
* WSL2 (Ubuntu)
* VS Code
* LaTeX Workshop extension
* latexmk
* biber
* Git + GitHub

The paper is developed completely inside the Linux WSL environment.

---

# 1. Install WSL2

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

Restart Windows after installation.

---

# 2. Install Ubuntu

List available distributions:

```powershell
wsl --list --online
```

Install Ubuntu:

```powershell
wsl --install -d Ubuntu
```

Restart Windows again if required.

---

# 3. Verify WSL Installation

Open PowerShell:

```powershell
wsl --status
```

Verify installed distributions:

```powershell
wsl -l -v
```

Expected output:

```text
Ubuntu    Running    2
```

---

# 4. Open Ubuntu

Start Ubuntu from the Windows Start Menu.

Create your Linux username and password when requested.

---

# 5. Update Ubuntu Packages

Inside Ubuntu:

```bash
sudo apt update
sudo apt upgrade -y
```

---

# 6. Install Git

```bash
sudo apt install git -y
```

Verify installation:

```bash
git --version
```

---

# 7. Install Complete LaTeX Environment

Install LaTeX, latexmk and biber:

```bash
sudo apt install texlive-full latexmk biber -y
```

This may take some time because the full TeX distribution is large.

Verify installation:

```bash
pdflatex --version
latexmk --version
biber --version
```

---

# 8. Install VS Code

Download and install VS Code from:

https://code.visualstudio.com/

---

# 9. Install Required VS Code Extensions

Open VS Code and install:

## WSL Extension

Search for:

```text
WSL
```

Published by Microsoft.

---

## LaTeX Workshop Extension

Search for:

```text
LaTeX Workshop
```

Published by James Yu.

---

# 10. Configure GitHub SSH Access

Generate SSH key inside WSL:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Start SSH agent:

```bash
eval "$(ssh-agent -s)"
```

Add key:

```bash
ssh-add ~/.ssh/id_ed25519
```

Print public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output.

---

## Add SSH Key to GitHub

Open:

```text
GitHub → Settings → SSH and GPG Keys
```

Add a new SSH key and paste the copied public key.

---

## Test GitHub Connection

```bash
ssh -T git@github.com
```

Expected output:

```text
Hi USERNAME! You've successfully authenticated...
```

---

# 11. Clone the Repository

Inside Ubuntu:

```bash
cd ~
mkdir -p projects
cd projects
```

Clone repository:

```bash
git clone git@github.com:YOUR_USERNAME/YOUR_REPOSITORY.git
```

Enter project:

```bash
cd YOUR_REPOSITORY
```

---

# 12. Open the Repository in VS Code

Inside WSL:

```bash
code .
```

VS Code should open with:

```text
WSL: Ubuntu
```

shown in the bottom-left corner.

This means VS Code is connected correctly to the Linux environment.

---

# 13. Recommended File Locations

Recommended:

```text
/home/<username>/projects/
```

Avoid storing the LaTeX project inside:

```text
/mnt/c/
```

because WSL filesystem performance is significantly better inside Linux directories.

---

# 14. Build the Paper

## Option 1 — Build from VS Code

Use:

```text
Ctrl + Alt + B
```

LaTeX Workshop will use the configured `latexmk` build arguments.

---

## Option 2 — Build from Terminal

Inside the project folder:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

---

# 15. Build With Glossary and Bibliography

Because the project uses:

* glossary
* bibliography
* biber

the recommended build sequence is:

```bash
latexmk -pdf main.tex
makeglossaries main
biber main
latexmk -pdf main.tex
```

Generated output:

```text
main.pdf
```

---

# 16. Clean Build Files

Remove temporary LaTeX files:

```bash
latexmk -c
```

Remove all generated files including PDF:

```bash
latexmk -C
```

---

# 17. Git Workflow

## Check status

```bash
git status
```

---

## Add changes

```bash
git add .
```

---

## Commit changes

```bash
git commit -m "Update introduction chapter"
```

---

## Push changes

```bash
git push
```

---

# 18. Files That Should Be Committed

Commit these files:

```text
*.tex
references.bib
logo.png
README.md
.gitignore
figures/
.vscode/settings.json
```

---

# 19. Files That Should NOT Be Committed

Generated build artifacts:

```text
*.aux
*.log
*.out
*.toc
*.lof
*.lot
*.fls
*.fdb_latexmk
*.synctex.gz
*.bbl
*.bcf
*.blg
*.run.xml
*.acn
*.acr
*.alg
*.glg
*.glo
*.gls
*.ist
```

The generated PDF:

```text
main.pdf
```

may optionally be committed if desired.

---

# 20. Recommended Workflow

1. Pull latest changes
2. Modify chapter files
3. Build PDF locally
4. Check references and glossary
5. Commit changes
6. Push to GitHub

---

# 21. Useful Commands

## Pull latest changes

```bash
git pull
```

---

## Create new branch

```bash
git checkout -b feature/new-chapter
```

---

## Switch branch

```bash
git checkout main
```

---

## Rebuild completely

```bash
latexmk -C
latexmk -pdf main.tex
makeglossaries main
biber main
latexmk -pdf main.tex
```

---

# 22. Notes

* The project uses IEEE bibliography style.
* Glossary references are created using `\gls{}` commands.
* Every chapter is stored in a separate `.tex` file.
* Figures should be stored in the `figures/` directory.
* The project is optimized for development inside WSL2.
