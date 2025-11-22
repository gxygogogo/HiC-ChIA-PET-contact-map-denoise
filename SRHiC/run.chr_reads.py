file_HiC   = "/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_tmp/WT_SMC1A/GM12878_WT_ChIAPET_SMC1A.remove_Diagonal.txt"
chr        = '16'
file_reads = "/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_tmp/WT_SMC1A/chr16_remove_Diagonal.reads"


fileIn  = open(file_HiC)
fileOut = open(file_reads, "w")

for line in fileIn:
    items = line.split("\t")
    if items[1] == chr and items[5] == chr:
       print(items)
       outstr = items[2] + " " + items[6] + "\n"
       fileOut.write(outstr)

fileIn.close()
fileOut.close()
