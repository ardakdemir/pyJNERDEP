#!/bin/bash

langs=(czech hungarian japanese english finnish  turkish)
word_vec_prefs=("cs" "hu" "jp" "en" "fi" "tr")
domains=("movie" "twitter")

for i in 0 1 2 3 4 5
do
  lang=${langs[$i]}
  for j in 0 1
  do
    domain=${domains[$j]}
    echo "Language: "${lang}" Domain: "${domain}
    file_path="../../datasets/sa_"${domain}"_"${lang}"-train.json"
    if [ -f ${file_path} ]
    then
      echo $file_path" "${lang}" "${domain}
    fi
  done
done