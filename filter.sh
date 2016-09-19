#!/bin/bash

cat $1 | grep "<abstract>"       \
       | grep "。"               \
       | sed "s/<abstract>//"    \
       | sed "s/<\/abstract>//"  \
       | grep -v "|"             \
       | grep -v "\*"            \
       | grep -v "『"            \
       | grep -v "』"            \
       | grep -v "「"            \
       | grep -v "」"            \
       | sed "s/[（].*[）$]//"   \
       | sed "s/[\(].*[\)$]//"   \
       | grep -v ")"             \
       | grep -v "("             \
       | grep -v "（"            \
       | grep -v "）"            \
> $2
