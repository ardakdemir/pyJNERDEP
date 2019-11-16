import parser.conll_ud_eval as ud_eval
import logging
dep_dict = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}
ner_dict = {"token": 0 , "morp": 1 ,"truth":2, "ner_tag":3}
dicts = {"dep": dep_dict, "ner":ner_dict}




def convert2IOB2new(input_file, output_file):
    a = open(input_file,encoding="utf-8").readlines()
    out = open(output_file,"w",encoding="utf-8")
    prev_tag_1 = "O"
    prev_tag_2 = "O"
    for line in a:
        if len(line)>2:
            ls = line.split()
            line = ls
            new_line = ""
            if ls[-2]!="O":
                if ls[-2][2:]!=prev_tag_2:
                    new_line = line[:-2] + ["B-"+ls[-2][2:]]
                else:
                    new_line = line[:-2] + ["I-"+ls[-2][2:]]
                if ls[-1]!="O":
                    if ls[-1][2:]!=prev_tag_1:
                        new_line = new_line + ["B-"+ls[-1][2:]]
                    else:
                        new_line = new_line + [ls[-1]]
                else:
                    new_line = new_line + [ls[-1]]
            elif ls[-1]!="O":
                new_line = line[:-1]
                if ls[-1][2:]!=prev_tag_1:
                    new_line = new_line + ["B-"+ls[-1][2:]]
                else:
                    new_line = new_line + [ls[-1]]
            else:
                new_line = line
            out.write("{}\n".format("\t".join([l for l in new_line])))
            prev_tag_1 = ls[-1][2:]
            prev_tag_2 = ls[-2][2:]
        else:
            prev_tag = "O"
            out.write("\n")


def convert2IOB2(input_file, output_file):
    a = open(input_file,encoding="utf-8").readlines()
    out = open(output_file,"w",encoding="utf-8")
    prev_tag = "O"
    for line in a:
        if len(line)>2:
            ls = line.split()
            line = ls
            if ls[-1]!="O":
                if ls[-1][2:]!=prev_tag:
                    new_line = line[:-1] + ["B-"+ls[-1][2:]]
                else:
                    new_line = line[:-1] + ["I-"+ls[-1][2:]]
            else:
                new_line = line
            out.write("{}\n".format("\t".join([l for l in new_line])))
            prev_tag = ls[-1][2:]
        else:
            prev_tag = "O"
            out.write("\n")



def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for UD parser scorer. """
    evaluation = ud_scores( system_conllu_file, gold_conllu_file)
    el = evaluation['LAS']
    uas = evaluation['UAS']
    p = el.precision
    r = el.recall
    f = el.f1
    uas_f1 = uas.f1
    logging.info('LAS : {}  and UAS : {} '.format(f,uas_f1))
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['LAS', 'MLAS', 'BLEX']]
        print("LAS\tMLAS\tBLEX")
        print("{:.2f}\t{:.2f}\t{:.2f}".format(*scores))
    return p, r, f, uas_f1



def extract_pos_from_morp(pos_dict,word, morp, unk_ind):
    morp = morp.split("+")
    morp.reverse()
    found=False
    for x in morp:
        if x in pos_dict:
            found = True
            return pos_dict[x]
        elif x[:-1] in pos_dict:
            found = True
            return pos_dict[x[:-1]]
        if x == "*UNKNOWN*":
            found = True
            return pos_dict[unk_ind]
        elif x == "Num":
            return pos_dict['NNum']
        elif x == "Adv":
            return pos_dict['Adverb']
        elif x == "PersP":
            return pos_dict['Pers'] 
            found = True
        elif word=="km" or word=="m":
            return pos_dict['Noun']
        elif word=='satın':
            return pos_dict['Adv']
            found = True
    if not found:
        print(f[i-2])
        print(f[i-1])
        print(line)
        print(f[i+1])
        print(f[i+2])
        c+=1
    return unk_ind       


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
    return posdict,c       



def conll_writer(file_name, content, field_names, task_name,verbose=False):
    out = open(file_name,'w',encoding = 'utf-8')
    task_dict = dicts[task_name]
    if verbose:
        out.write("{}\n".format("\t".join([k for k in task_dict])))
    init = ["-" for i in range(len(task_dict))]
    for sent in content:
        for id,tok in enumerate(sent):
            for i,f in enumerate(field_names):
                init[task_dict[f]] = tok[i]
                if type(tok[i])==int:
                    init[task_dict[f]]=str(tok[i])
            if task_name == 'dep':
                init[0] =  str(id+1)
            out.write("{}\n".format("\t".join(init)))
        out.write("\n")
    out.close()

def ud_scores(system_conllu_file, gold_conllu_file):
    cropped = 'cropped'
    get_conll_file_for_eval(gold_conllu_file,cropped)
    gold_ud = ud_eval.load_conllu_file(cropped)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)
    #print(evaluation)
    return evaluation

def get_conll_file_for_eval(conll_u,out_file="cropped"):
    conll = open(conll_u,encoding='utf-8').readlines()
    out = open(out_file,"w",encoding="utf-8")
    for line in conll:
        if line.startswith("#"):
            continue
        elif line.split()==[]:
            out.write("\n")
        elif "-" in line.split()[0]:
            continue
        else:
            out.write(line)
    out.close()
def sort_dataset(dataset,desc=True, sort = True):
    
    idx = [i for i in range(len(dataset))]
    if not sort:
        return dataset, idx
    zipped = list(zip(dataset,idx))
    zipped.sort(key = lambda x : len(x[0]))
    if desc:
        zipped.reverse()
    dataset, orig_idx = list(zip(*zipped))
    return dataset, orig_idx
def unsort_dataset(dataset,orig_idx):
    zipped = list(zip(dataset,orig_idx))
    zipped.sort(key = lambda x : x[1])
    dataset , _ = list(zip(*(zipped)))
    return dataset
if __name__ == "__main__":
    convert2IOB2_new('joint_ner_out.txt','iobnerout')
