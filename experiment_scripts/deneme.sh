#!/bin/bash
file_path="../../datasets/"${lang}"_"${domain}"_train.json"
if [ -f ${file_path} ]
then
  echo $file_path
fi