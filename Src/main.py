from random import randint



def choose_random_categories(num):
    used = []
    ret = []
    while len(ret) < num:
        sel = randint(1, 20) - 1
        if sel not in used:
            used.append(sel)
            ret.append(sel)
    return ret
