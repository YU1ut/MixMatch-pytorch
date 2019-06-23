import os
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        print (path+" created")
        os.makedirs(path)
        return True
    else:
        print (path+' existed')
        return False
def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text