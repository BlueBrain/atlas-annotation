#!/usr/bin/env bash


for f in $(find -E . -regex '^\./(_|ants|ml).*\.py$' | sort)
do
  echo "================================"
  echo $f
  python "$f"
done
