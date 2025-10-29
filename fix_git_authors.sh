#!/bin/bash
# Script to rewrite git history and change all commits to use the correct author
# WARNING: This rewrites history. Use with caution and backup first!
# After running, you'll need to force push: git push --force --all

CORRECT_NAME="abhiramavarma"
CORRECT_EMAIL="abhiramavarma@gmail.com"

echo "Rewriting git history to set all commits to: $CORRECT_NAME <$CORRECT_EMAIL>"
echo "This will modify all commits. Press Ctrl+C to cancel, or Enter to continue..."
read

git filter-branch --force --env-filter "
OLD_EMAILS=('amanvirparhar@gmail.com' '46307450+amanvirparhar@users.noreply.github.com')
CORRECT_NAME='$CORRECT_NAME'
CORRECT_EMAIL='$CORRECT_EMAIL'

for OLD_EMAIL in \"\${OLD_EMAILS[@]}\"; do
    if [ \"\$GIT_COMMITTER_EMAIL\" = \"\$OLD_EMAIL\" ]; then
        export GIT_COMMITTER_NAME=\"\$CORRECT_NAME\"
        export GIT_COMMITTER_EMAIL=\"\$CORRECT_EMAIL\"
    fi
    if [ \"\$GIT_AUTHOR_EMAIL\" = \"\$OLD_EMAIL\" ]; then
        export GIT_AUTHOR_NAME=\"\$CORRECT_NAME\"
        export GIT_AUTHOR_EMAIL=\"\$CORRECT_EMAIL\"
    fi
done
" --tag-name-filter cat -- --branches --tags

echo ""
echo "History rewritten! To verify: git log --format='%an %ae' | sort -u"
echo "To push changes: git push --force --all"

