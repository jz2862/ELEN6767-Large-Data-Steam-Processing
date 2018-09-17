import sys


with open('easy.txt','r') as f:
    flist = f.readlines()
    flist = [i.strip() for i in flist]
    flist = [('_').join(i.split(' ')) for i in flist]
    print flist
# with open('easy0.txt','w') as f:
#     orig_stdout = sys.stdout
#     sys.stdout = f
#     for i in flist:
#         print i
#     sys.stdout = orig_stdout
with open('easy0.txt', 'r') as f:
    easyname = f.readlines()
    easyname = [i.strip() for i in easyname]
    print easyname
