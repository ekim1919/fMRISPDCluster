#!/bin/sh
for c1 in 'spd',60 'rspd',10
do IFS=","; set -- ${c1};
    for c2 in .1 .01 .001
    do
        for c3 in 2 5 7
	    do
            for c4 in 30 50 60
            do
                python examples/SPDdemo.py -model "${1}" -epochs 50 -lr "${c2}" -hs ${c3} -bs ${2} -ws ${c4} -use_cuda 
            done
        done
    done
done

