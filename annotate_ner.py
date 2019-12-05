ROOT= "[ROOT]"
ROOT_IND = 1

def read_conllu(file_name, cols = ['word','upos','head','deprel']):
    """
        Reads a conllu file and generates the vocabularies
    """
    assert file_name.endswith("conllu"), "File must a .conllu type"
    file = open(file_name, encoding = "utf-8").read().rstrip().split("\n")
    dataset = []
    sentence = []
    tok2ind = {ROOT : ROOT_IND}
    pos2ind = {ROOT : ROOT_IND,"[UNK]":0}
    dep2ind = {ROOT : ROOT_IND}
    pos_counter = Counter()
    total_word_size = 0
    root = [[ROOT for _ in range(len(cols))]]
    for line in file:
        if line.startswith("#"):
            continue
        elif line=="":
            sentence = root + sentence
            dataset.append(sentence)
            sentence = []
        else:
            line = line.split("\t")
            if "-" in line[0]: #skip expanded words
                continue
            total_word_size+=1
            sentence.append([line[FIELD_TO_IDX[x.lower()]] for x in cols])
            pos_counter.update([line[4]])
            if line[1] not in tok2ind:
                tok2ind[line[1]] = len(tok2ind)
            if line[4] not in pos2ind:
                pos2ind[line[4]] = len(pos2ind)
            if line[7] not in dep2ind:
                dep2ind[line[7]] = len(dep2ind)
    if len(sentence):
        sentence = root + sentence
        dataset.append(sentence)
    return pos2ind,pos_counter

def extract_pos_ner(pos_dict, ner_file):
    posdict = {}
    f = open(ner_file,encoding='utf-8').read().rstrip().split("\n")
    c = 0
    for i, line in enumerate(f):
        if line=="":
            continue
        ls = line.split()
        morp = ls[1].split("+")
        morp.reverse()
        found=False
        for x in morp:
            if x in pos_dict:
                found = True
                if x not in posdict:
                    posdict[x]=len(posdict)
                break
            elif x[:-1] in pos_dict:
                if x[:-1] not in posdict:
                    posdict[x] = len(posdict)
                found = True
            if x == "*UNKNOWN*":
                found = True
            elif x == "Num" or x=="Adv" or x=="PersP":
                found = True
            elif ls[0]=="km" or ls[0]=="satın" or ls[0]=="m":
                found = True
        if not found:
            print(f[i-2])
            print(f[i-1])
            print(line)
            print(f[i+1])
            print(f[i+2])
            c+=1
    assert c==0, "Un mapped elements exist!!"
    return posdict,c       








