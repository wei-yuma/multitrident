global glb_dict
glb_dict = {}

def addItem(key, value):
    glb_dict[key]=value

def getItem(key):
    return glb_dict[key]