dic = {}
dic2 = {}
with open("./data/train.csv",'r') as f:
    lines = f.readlines()
    for line in lines:
        c = line.split(',')[0]
        if c in dic:
            dic[c] += 1
        else:
            dic[c] = 1
    for i in sorted(dic):
        dic2[i] = dic[i]
    print(dic2)
