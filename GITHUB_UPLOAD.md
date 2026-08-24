# Publish the APA report and updated laboratory page

The update places the website at the repository root and the report in `report/`.

## Files to copy into the local repository

- `index.html` → repository root
- `report/Los_Angeles_Temperature_Forecasting_Laboratory_APA_Report.pdf`
- `report/Los_Angeles_Temperature_Forecasting_Laboratory_APA_Report.docx`
- `.github/workflows/pages.yml`

## Git Bash commands

```bash
cd "/c/Users/salem/GitHub/Los-Angeles-Daily-Temperature-High"

git status
git add index.html report .github/workflows/pages.yml
git commit -m "Add APA report and link it from forecast laboratory"
git pull --rebase origin main
git push origin main
git status
```

After GitHub Actions completes, open:

- Site: https://salemmorelli1.github.io/Los-Angeles-Daily-Temperature-High/
- Report: https://salemmorelli1.github.io/Los-Angeles-Daily-Temperature-High/report/Los_Angeles_Temperature_Forecasting_Laboratory_APA_Report.pdf

If Git reports `nothing to commit`, verify that the four update files were copied into the local repository paths listed above before rerunning `git add`.
