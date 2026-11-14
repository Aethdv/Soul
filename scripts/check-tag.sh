#!/usr/bin/env bash
#
# Refuses a release build unless HEAD is the commit v$1 points at.
#
#   scripts/check-tag.sh <version>

version=${1:?usage: check-tag.sh <version>}
head=$(git rev-parse -q --verify HEAD 2>/dev/null)
tag=$(git rev-parse -q --verify "v$version^{commit}" 2>/dev/null)

if [[ -n $head && $head == "$tag" ]]; then
    exit 0
fi

alarm='\033[38;2;225;89;91m'
ivory='\033[38;2;246;238;218m'
reset='\033[0m'

printf "%bHEAD is not tagged v%s.%b\n" "$alarm" "$version" "$reset" >&2
printf "Tag it first:  %bgit tag v%s%b\n" "$ivory" "$version" "$reset" >&2
exit 1
