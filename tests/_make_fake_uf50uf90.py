alphabet = 'ARNDCQEGHILKMFPSTWYV'

cluster_mapping = {}
with open('Fake50.fasta', 'w') as fd:
    for length in range(50, 1000, 50):
        for base in alphabet:
            fd.write(f">UniRef50_{base}_{length}\n")
            fd.write(base * length)
            fd.write("\n")
            cluster_mapping[f"UniRef50_{base}_{length}"] = []


with open('Fake90.fasta', 'w') as fd:
    num_clusters = 10
    for length in range(50, 1000, 50):
        for base in alphabet:
            for i in range(num_clusters):
                fd.write(f">UniRef90_{base}_{length}-{i}\n")
                fd.write(base * (length + i))
                fd.write("\n")
                cluster_mapping[f"UniRef50_{base}_{length}"].append(f"UniRef90_{base}_{length}-{i}")
            num_clusters += 1


with open("mapping.tsv", 'w') as fd:
    fd.write("dumbheader\n")
    for key, values in cluster_mapping.items():
        str_ = key + "\t" + ",".join(values)
        fd.write(f"{str_}\n")
