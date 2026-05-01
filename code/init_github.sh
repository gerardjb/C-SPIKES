#!/bin/bash
git remote add github https://github.com/gerardjb/C-SPIKES
git fetch github
git config pull.rebase false
echo "GitHub remote attached and fetched!"
echo "You can now sync to the code ocean repo with 'git pull github code-ocean-manuscript'"
echo "You may need the --allow-unrelated-histories flag if changes need to be merged on master"
echo "In that case, use 'git checkout --theirs...' to resolve any differences, then add and commit"