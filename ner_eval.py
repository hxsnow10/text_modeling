# encoding=utf-8
import sys

def ner_eval(test_path, gold_path):
    P,T,PT=0,0,0
    for test, gold in zip(open(test_path), open(gold_path)):
        iP=set([x.split('/')[0] for x in test.split(' ') if '/' in x])
        iT=set([x.split('/')[0] for x in gold.split(' ') if '/' in x])
        if  iP!=iT: 
            print "test:", test
            print "gold:", gold
        iPT=iP&iT
        P+=len(iP)
        T+=len(iT)
        PT+=len(iPT)
    p,r = PT*1.0/P, PT*1.0/T
    f = 2*p*r/(p+r)
    return {"f1":f,"p":p,"r":r} 

if __name__=="__main__":
    print ner_eval(sys.argv[1], sys.argv[2]) 
