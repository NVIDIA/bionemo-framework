mmseqs createdb *.fa allSeqs; 

for i in *.fa; 
do 
    j=${i%.fa}; 
    mmseqs createdb $i ${j}Seqs; 
    mmseqs linclust ${j}Seqs ${j}SeqClust ${j}tmp --min-seq-id 0.5 -c 0.9 --cov-mode 5;
    mmseqs createsubdb ${j}SeqClust ${j}Seqs ${j}SeqClust_rep;
    mmseqs search ${j}SeqClust_rep allSeqs ${j}ResultDB ${j}tmp --max-accept 1000000 --max-rejected 1000000 --min-seq-id 0.3 --cov-mode 5 -c 0.8 --max-seq-id 0.95 --max-seqs 500 -s 4;
    mmseqs result2msa ${j}SeqClust_rep allSeqs ${j}ResultDB ${j}SeqClust_rep.msa.a3m --msa-format-mode 5;
done