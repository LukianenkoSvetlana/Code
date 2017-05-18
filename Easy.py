
def zeroes(n):
    z = []
    j = 1
    while j <= n:
        z.append(0)
        j += 1
    return z


def clear(w):
    point = 0
    w1 = ''
    for i in range(len(w)):
        if (w[i] == '.'):
            point = 1
        else:
            num = ord(w[i])
            if not ((num < 48) or (57 < num < 65) or (90 < num < 97) or (122 < num < 167) or (167 < num < 1040)
                   or (num > 1103)):
                w1 = w1 + w[i]
    return w1, point


def is_number(w):
    for i in w:
        if i not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return 0
    return 1


def last(m_list):
    x = len(m_list)
    return m_list[x - 1]


def to_str(n_list):
    str_m = '['
    i = 0
    while i < len(n_list):
        str_m = str_m + str(n_list[i]) + '  '
        i += 1
    str_m += ']'
    return str_m


def check_first_for_collocation (pos):
    pm = ('NOUN')

    if pos in pm:
        return 1
    return 0


def check_second_for_collocation (pos):
    pm = ('NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'NUMR', 'ADVB')

    if pos in pm:
        return 1
    return 0


def check_waste(w, stop_words):
    if w in stop_words:
        return 1
    return 0

# files = os.listdir(path="Input/")
