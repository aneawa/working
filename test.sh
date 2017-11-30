#!/bin/sh
for i in 0.1 0.2 0.3 0.4 0.5
  do
    for j in 1 2 3
      do
        python test_isolationforest.py /home/anegawa/Dropbox/cover.mat $i $j 
      done
  done
echo おわおわり
