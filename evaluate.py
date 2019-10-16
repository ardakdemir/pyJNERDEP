

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
