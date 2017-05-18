import math


def dice(f1, f2, ft):
    return 2 * ft / (f1 + f2)


def logdice(f1, f2, ft):
    return 14 + math.log2(dice(f1, f2, ft))


def mi(f1, f2, ft, wt):
    return math.log2(ft * wt / f1 / f2)


def mi3(f1, f2, ft, wt):
    return math.log2(pow(ft, 3) * wt / f1 / f2)


def mi_log(f1, f2, ft, wt):
    return mi(f1, f2, ft, wt) * math.log2(ft + 1)


def t_score(f1, f2, ft, wt):
    return (ft - f1 * f2 / wt) / math.sqrt(ft)


def tf_idf(tn, td, d):
    return tn * math.log2(d / td)


def tv(tn, col_in_doc, d):
    sum = 0
    for doc in col_in_doc:
        sum += col_in_doc[doc] - tn/d
    return sum


def weirdness(tn1, size1, tn2, size2):
    if tn2 > 0:
        return (tn1 / size1) / (tn2 / size2)
    else:
        return tn1 / size1


def relevance(tn1, tn2, df1):
    if tn2 > 0:
        return 1 - 1 / (math.log2(2 + tn1 * df1 / tn2))
    else:
        return 1 - 1 / (math.log2(2 + tn1 * df1 / 0.01))


def llh(tn1, size1, tn2, size2):
    tf1 = size1 * (tn1 + tn2) / (size1 + size2)
    tf2 = size2 * (tn1 + tn2) / (size1 + size2)
    if tn2 > 0:
        return 2 * (tn1 * math.log2(tn1 / tf1) + tn2 * math.log2(tn2 / tf2))
    else:
        return 2 * tn1 * math.log2(tn1 / tf1)