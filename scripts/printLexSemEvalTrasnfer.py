# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:02:19 2016

@author: admin
"""


def is_ascii(s):
    return all(ord(c) < 128 for c in s)

lex_file='/Users/admin/Dropbox/THESIS/IJCAI2016/Tables/SemEvalLex.csv'    

f=open(lex_file, "rb")
output_file='/Users/admin/Dropbox/THESIS/IJCAI2016/Tables/SemEvalLexClean.csv'

out=open(output_file,"w")
             
        


for line in f.readlines():
    parts=line.strip().split('\t')
    
    if(is_ascii(line) and len(parts)==3):
        out.write(parts[0]+'\t'+parts[1]+'\t'+parts[2]+'\n')   

f.close()
out.close()