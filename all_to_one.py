import os

out_file = "all_t_3.csv"
fo = open(out_file, 'w')
fo.write("Word1; Word2; Position; Total Number; Dice; logDice; MI; MI3; MIlog; t-score; tf-idf; TV; c-value; FLR; Total Number in contrast collection; Weirdness; Relevance; LogLikelihood\n")

dir_name = "Output/3"
files = os.listdir(dir_name)
for f in files:
    f_name = dir_name + "/" + f
    ff = open(f_name)
    source = [row.strip().split(';') for row in ff]
    if len(source) > 1:
        print(source)
        i = 1
        str = ""
        while i < len(source):
            for elem in source[i]:
                str += elem + '; '
            str = str[:len(str) - 2] + '\n'
            fo.write(str)
            str = ""
            i += 1
