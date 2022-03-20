EPS = "<eps>"
CHARS = list("abcdefghijklmnopqrstuvwxyz")
INFINITY = 1000000000

def format_arc(src, dest, ilabel, olabel, weight=0):
    return (str(src) + " " + str(dest) + " " + str(ilabel) + " " + str(olabel) + " " + str(weight))
