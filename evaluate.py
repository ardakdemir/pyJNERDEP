

class Evaluate():
    def __init__(self,task_name):
        self.task_name = task_name


    def countNonZeroMatch(self, preds, labels, zero_ind):
        c = 0
        t = 0
        p_tot = 0
        p = preds
        for p_,l_ in zip(p,labels):
            lx = l_
            px = p_
            if lx==px!=0:
                c+=1
            if lx!=0:
                t +=1
            if px!=0:
                p_tot+=1
        return c,t,p_tot
    
    def conll_eval(self,pred_file):
        true_pos = 1
        preds = 1
        truths = 1
        pred_file = open(pred_file, encoding = 'utf-8').readlines()
        for line in pred_file:
            ls = line.split()
            if len(ls)>2:
                if ls[-1]!="O" and  ls[-2]!="O":
                    if ls[-1][2:]==ls[-2][2:]:
                        true_pos +=1
                        preds +=1
                        truths +=1
                    else:
                        preds +=1
                        truths +=1
                elif ls[-2]=="O" and ls[-1]!="O":
                    preds +=1
                elif ls[-1]=="O" and ls[-2]!="O":
                    truths +=1

        prec = true_pos/preds
        rec = true_pos/truths
        f1 = 2 * (prec*rec) / (prec+rec)
        return prec, rec, f1 
    def f_1(self,preds,labels):
        c, t, p_tot = self.countNonZeroMatch(preds, labels, 0)
        rec = 0
        pre = 0
        if t>0:
            rec = c/t
        if p_tot > 0:
            pre = c/p_tot
        #print("Precision : ", pre , "  Recall : ", rec)
        return c, p_tot, t

