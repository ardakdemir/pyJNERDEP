import parser.conll_ud_eval as ud_eval
import logging
dep_dict = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}
ner_dict = {"token": 0 , "morp": 1 ,"truth":2, "ner_tag":3}
dicts = {"dep": dep_dict, "ner":ner_dict}




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
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['LAS', 'MLAS', 'BLEX']]
        print("LAS\tMLAS\tBLEX")
        print("{:.2f}\t{:.2f}\t{:.2f}".format(*scores))
    return p, r, f




def conll_writer(file_name, content, field_names, task_name,verbose=False):
    out = open(file_name,'w',encoding = 'utf-8')
    task_dict = dicts[task_name]
    if verbose:
        out.write("{}\n".format("\t".join([k for k in task_dict])))
    init = ["-" for i in range(len(task_dict))]
    for sent in content:
        logging.info(sent)
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

    out_name = 'dependency_out.conllu'
