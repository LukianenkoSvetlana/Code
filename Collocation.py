import math
import os
import pickle

import Easy
import pymorphy2

from Code import Stat_func

morph = pymorphy2.MorphAnalyzer()
spector = 2


class Statistic:
    word_stat = {}
    collocations = []
    file_num = 0

    def import_dir(self, dirname, collection):
        files = os.listdir(dirname)
        for f in files:
            fname = dirname + '/' + f
            print(fname)
            Statistic.file_to_words(self, fname, 1, collection)
            self.file_num += 1

    def file_to_words(self, filename, ifdir, collection):
        stop_words = []
        with open('waste.txt') as f:
            stop_words = f.read().splitlines()
        f = open(filename)
        if not ifdir:
            name_f = "Input" + collection + '/' + filename
        else:
            s_end = filename.find("/")
            while s_end != -1:
                filename = filename[0:s_end] + '_' + filename[s_end + 1:]
                s_end = filename.find("/")
            name_f = "Input" + collection + '/' + filename
        s_end = name_f.find(".txt")
        name_f = name_f[0:s_end]
        self.file_num += 1

        file_list = []
        sentence = 0

        for line in f:
            word = []
            l = line.split()
            # print(l)
            for i in range(len(l)):
                [w1, point] = Easy.clear(l[i])
                # if sentence % 50 == 0:
                #     print(w1, end=', ')
                if w1 != '':
                    if Easy.is_number(w1):
                        pm = ['#ЧИСЛО#', 'NUMR']
                    else:
                        p = morph.parse(w1)[0]
                        if p.tag.POS == 'NUMR':
                            pm = ['#ЧИСЛО#', 'NUMR']
                        else:
                            pm = [p.normal_form, str(p.tag.POS)]
                    word.append(pm)
                    if point:
                        word.append(['#ТОЧКА#', 'POINT'])
                    file_list.append(pm)
                    # new_f.write(str(pm) + '\n')

                    if pm[0] in self.word_stat:
                        self.word_stat[pm[0]] += 1
                    else:
                        self.word_stat[pm[0]] = 1
            # if sentence % 50 == 0:
            #     print('\n')
            for i in range(len(word)):
                r = - spector
                if word[i][0] == '#ТОЧКА#':
                    sentence += 1
                else:
                    while r <= spector:
                        if (i + r >= 0) & (i + r < len(word)) & (r != 0):
                            j = 0
                            find = 0
                            tmp = (self.file_num, sentence)
                            if word[i + r][0] != '#ТОЧКА#':
                                while (find == 0) & (j < len(self.collocations)):
                                    if self.collocations[j].is_equal(word[i][0], word[i + r][0], word[i][1],
                                                                     word[i + r][1]):
                                        find = 1
                                        self.collocations[j].add(r + spector, tmp)
                                    j += 1
                                # if (find == 0):
                                #     print(word[i][1], "  ", word[i + r][1])
                                if (find == 0) & (Easy.check_first_for_collocation(word[i][1])) & (
                                        Easy.check_second_for_collocation(word[i + r][1])) & (
                                        not (Easy.check_waste(word[i + r][1], stop_words))):
                                    col = Collocation(word[i][0], word[i + r][0], word[i][1], word[i + r][1],
                                                      r + spector, tmp)
                                    self.collocations.append(col)
                        r += 1
        f.close()

        print("Import of " + filename + " done")

        with open(name_f, 'wb') as f:
            pickle.dump(file_list, f)

    def find_collocations(self, word):
        result = []

        p = morph.parse(word)[0]
        word = p.normal_form

        i = 0
        while i < len(self.collocations):
            if self.collocations[i].is_word(word):
                result.append(self.collocations[i])
            i += 1
        i = 0
        while i < len(result):
            print(result[i])
            i += 1
        print(i)

    def save_word_stat(self, collection):
        filename = "Statistic/word_stat" + collection
        with open(filename, 'wb') as file_ws:
            pickle.dump(self.word_stat, file_ws)

    def save_collocations(self, collection):
        filename = "Statistic/collocation" + collection
        with open(filename, 'wb') as file_col:
            pickle.dump(self.collocations, file_col)

    def import_word_stat(self, collection):
        filename = "Statistic/word_stat" + collection
        if os.path.exists(filename):
            with open(filename, 'rb') as file_ws:
                self.word_stat = pickle.load(file_ws)
        if os.path.exists('Input' + collection):
            files = os.listdir('Input' + collection)
            self.file_num = len(files)
        else:
            self.file_num = 0

    def import_collocation(self, collection):
        filename = "Statistic/collocation" + collection
        if os.path.exists(filename):
            with open(filename, 'rb') as file_col:
                self.collocations = pickle.load(file_col)

    def col_measures(self, word, collection):
        stop_words = []
        with open('waste.txt') as f:
            stop_words = f.read().splitlines()

        # print(stop_words)

        slash = word.find('/')
        if slash != -1:
            word = word[:slash] + word[slash:]

        cl_word, _ = Easy.clear(word)

        filename = "Output/" + cl_word + ".csv"

        space = word.find(" ")
        if space != -1:
            #     словосочетание из двух и более слов
            w1 = word[:space]
            w2 = word[space:]

            space = w2.find(" ")
            if space == -1:
                print("ERROR: More then two words in termin -- ", word, ". Can not count measures")
                return

            w1, _ = Easy.clear(w1)
            w2, _ = Easy.clear(w2)
            p = morph.parse(w1)[0]
            w1 = p.normal_form
            p = morph.parse(w2)[0]
            w2 = p.normal_form

            print('two words -- ', word)
            word = w1 + ' ' + w2

            i = 0
            find = 0
            while i < len(self.collocations):
                if self.collocations[i].word1 == w1 and self.collocations[i].word2 == w2:
                    find = 1
                    count_pair = self.collocations[i].number[3]
                    break
                if self.collocations[i].word1 == w2 and self.collocations[i].word2 == w1:
                    find = 2
                    count_pair = self.collocations[i].number[1]
                    break
                i += 1
            if find == 0:
                print("ERROR: ", word, " -- termin not found. Can not count measures")
                return

            # print(self.collocations[i])

            words = self.build_collocations(i, find, word)

            i = 0
            find = 0
            while i < len(collection.collocations):
                if collection.collocations[i].word1 == w1 and collection.collocations[i].word2 == w2:
                    find = 1
                    break
                if collection.collocations[i].word1 == w2 and collection.collocations[i].word2 == w1:
                    find = 2
                    break
                i += 1
            if find != 0:
                contrast_words = collection.build_collocations(i, find, word)

            f = open(filename, 'w')

            f.write(
                "Word1; Word2; Position; Total Number; Dice; logDice; MI; MI3; MIlog; t-score; tf-idf; TV; c-value; FLR; Total Number in contrast collection; Weirdness; Relevance; LogLikelihood\n")

            i = 0
            while i < len(words):
                if (words[i].total_near() > 0) and not (Easy.check_waste(words[i].word2, stop_words)):
                    c2 = self.word_stat[words[i].word2]
                    # print(words[i])
                    outs = words[i].measures(count_pair, c2, len(self.word_stat), self.file_num)
                    outs += ";0;0;"
                    if find != 0:
                        outs += self.contrast_measures_2(words[i], collection, contrast_words)
                    else:
                        outs += "0;0;0;0"
                    outs += "\n"
                    # print(outs)
                    f.write(outs)
                i += 1
        else:
            #     одно слово
            print("One word -- ", word)
            p = morph.parse(word)[0]
            word = p.normal_form
            f = open(filename, 'w')

            f.write(
                "Word1; Word2; Position; Total Number; Dice; logDice; MI; MI3; MIlog; t-score; tf-idf; TV; c-value; FLR; Total Number in contrast collection; Weirdness; Relevance; LogLikelihood\n")

            i = 0
            found = 0
            # print(len(self.collocations))
            while i < len(self.collocations):
                if self.collocations[i].word1 == word:
                    print(self.collocations[i])
                if self.collocations[i].is_word(word) and (self.collocations[i].total_near() > 0) and not (
                        Easy.check_waste(self.collocations[i].word2, stop_words)):
                    print(self.collocations[i])
                    found = 1
                    w1 = self.word_stat[word]
                    w2 = self.word_stat[self.collocations[i].word2]
                    outs = self.collocations[i].measures(w1, w2, len(self.word_stat), self.file_num)
                    outs += ";"
                    outs += self.context_measures(self.collocations[i])
                    outs += ";"
                    outs += self.contrast_measures(self.collocations[i], collection)
                    outs += "\n"
                    f.write(outs)
                i += 1
            # if (found == 0):
            #     print(i)

    def build_collocations(self, i, find, word):
        words = []
        k = 0
        while k < len(self.collocations):
            if self.collocations[k].is_word(self.collocations[i].word1):
                if find == 2:
                    for j in self.collocations[k].sentences[0]:
                        if j in self.collocations[i].sentences[1]:
                            # print(self.collocations[k].word2)
                            c = 0
                            f = 0
                            while c < len(words):
                                if self.collocations[k].word2 == words[c].word2:
                                    words[c].add(1, j)
                                    f = 1
                                c += 1
                            if f == 0:
                                col = Collocation(word, self.collocations[k].word2, 'PL', '*', 1, j)
                                words.append(col)
                    for j in self.collocations[k].sentences[3]:
                        if j in self.collocations[i].sentences[1]:
                            # print(self.collocations[k].word2)
                            c = 0
                            f = 0
                            while c < len(words):
                                if self.collocations[k].word2 == words[c].word2:
                                    words[c].add(3, j)
                                    f = 1
                                c += 1
                            if f == 0:
                                col = Collocation(word, self.collocations[k].word2, 'PL', '*', 3, j)
                                words.append(col)
                if find == 1:
                    for j in self.collocations[k].sentences[1]:
                        if j in self.collocations[i].sentences[3]:
                            # print(self.collocations[k].word2)
                            c = 0
                            f = 0
                            while c < len(words):
                                if self.collocations[k].word2 == words[c].word2:
                                    words[c].add(1, j)
                                    f = 1
                                c += 1
                            if f == 0:
                                col = Collocation(word, self.collocations[k].word2, 'PL', '*', 1, j)
                                words.append(col)
                    for j in self.collocations[k].sentences[4]:
                        if j in self.collocations[i].sentences[3]:
                            # print(self.collocations[k].word2)
                            c = 0
                            f = 0
                            while c < len(words):
                                if self.collocations[k].word2 == words[c].word2:
                                    words[c].add(3, j)
                                    f = 1
                                c += 1
                            if f == 0:
                                col = Collocation(word, self.collocations[k].word2, 'PL', '*', 3, j)
                                words.append(col)
            k += 1
        return words

    def contrast_measures_2(self, col, collection, words):
        word1 = col.word1
        word2 = col.word2
        near_count = col.total_near()
        doc_count = col.total_documents()
        size = self.col_size()

        col_size = collection.col_size()

        # print("size = ", size, ";  col_size = ", col_size)

        i = 0
        find = 0
        col_count = 0
        col_doc = 0
        while (find == 0) & (i < len(words)):
            if words[i].is_word(word1) and words[i].word2 == word2:
                find = 1
                col_count = words[i].total_near()
                col_doc = words[i].total_documents()
            i += 1

        # print("near = ", near_count, ";  col_count = ", col_count)

        if col_count > 0:
            print(word1, " - ", word2, " : ", near_count, " ", col_count)
        weird = Stat_func.weirdness(near_count, size, col_count, col_size)
        relevance = Stat_func.relevance(near_count, col_count, doc_count)
        llh = Stat_func.llh(near_count, size, col_count, col_size)

        return "%f; %f; %f; %f" % (col_count, weird, relevance, llh)

    def contrast_measures(self, col, collection):
        word1 = col.word1
        word2 = col.word2
        near_count = col.total_near()
        doc_count = col.total_documents()
        size = self.col_size()

        [word1_count, word2_count, col_count, col_size, col_doc] = collection.param(word1, word2)

        if col_count > 0:
            print(word1, " - ", word2, " : ", near_count, " ", col_count)
        weird = Stat_func.weirdness(near_count, size, col_count, col_size)
        relevance = Stat_func.relevance(near_count, col_count, doc_count)
        llh = Stat_func.llh(near_count, size, col_count, col_size)

        return "%f; %f; %f; %f" % (col_count, weird, relevance, llh)

    def param(self, word1, word2):
        ws1 = self.word_s(word1)
        ws2 = self.word_s(word2)
        col_size = self.col_size()

        i = 0
        find = 0
        col_count = 0
        col_doc = 0
        while (find == 0) & (i < len(self.collocations)):
            if self.collocations[i].is_word(word1) and self.collocations[i].word2 == word2:
                find = 1
                col_count = self.collocations[i].total_near()
                col_doc = self.collocations[i].total_documents()
            i += 1
        return ws1, ws2, col_count, col_size, col_doc

    def tmp(self, word, collection):
        p = morph.parse(word)[0]
        word = p.normal_form

        i = 0
        # print(len(self.collocations))
        while i < len(self.collocations):
            # if (i % 50) == 0: print(i)
            if self.collocations[i].is_word(word) and self.collocations[i].total_near() > 0:
                print(self.collocations[i].word1, " -- ", self.collocations[i].word2, " : ",
                      self.contrast_measures(self.collocations[i], collection))
                # self.context_measures(self.collocations[i]))
            i += 1

    def context_measures(self, coloc):  # TODO ориентировано на spectre = 2
        # print("c-value for ", coloc.word1, coloc.word2)
        word_num = 0
        token_l = 0
        token_r = 0
        words = {}
        sum = 0
        i = 0
        if coloc.total_near() == 0:
            return -1
        # print(coloc.word1, " -- ", coloc.word2, " : ", coloc.sentences, "  ", coloc.sentences_near)
        while i < len(self.collocations):
            if self.collocations[i].is_word(
                    coloc.word1):  # and (self.collocations[i].word2 != coloc.word2)) or ((self.collocations[i].is_word(coloc.word2)) and (self.collocations[i].word2 != coloc.word1))
                for j in self.collocations[i].sentences[0]:
                    if j in coloc.sentences[1]:
                        # print(self.collocations[i].word2)
                        token_l += 1
                        if self.collocations[i].word2 in words:
                            words[self.collocations[i].word2] += 1
                        else:
                            words[self.collocations[i].word2] = 1
                            word_num += 1
                for j in self.collocations[i].sentences[1]:
                    if j in coloc.sentences[3]:
                        # print(self.collocations[i].word2)
                        token_l += 1
                        if self.collocations[i].word2 in words:
                            words[self.collocations[i].word2] += 1
                        else:
                            words[self.collocations[i].word2] = 1
                            word_num += 1
                for j in self.collocations[i].sentences[3]:
                    if j in coloc.sentences[1]:
                        # print(self.collocations[i].word2)
                        token_r += 1
                        if self.collocations[i].word2 in words:
                            words[self.collocations[i].word2] += 1
                        else:
                            words[self.collocations[i].word2] = 1
                            word_num += 1
                for j in self.collocations[i].sentences[4]:
                    if j in coloc.sentences[3]:
                        # print(self.collocations[i].word2)
                        token_r += 1
                        if self.collocations[i].word2 in words:
                            words[self.collocations[i].word2] += 1
                        else:
                            words[self.collocations[i].word2] = 1
                            word_num += 1
            i += 1
        for w in words.keys():
            sum += words[w]
        flr = coloc.total_near() * math.sqrt(token_r * token_l)
        # print(coloc.total_number(), "  ", sum, "  ", word_num)
        if word_num == 0:
            return "%f; %f" % (coloc.total_number(), flr)
        else:
            return "%f; %f" % ((coloc.total_number() - (sum / word_num)), flr)

    def word_s(self, word):
        return self.word_stat.get(word, 0)

    def col_size(self):
        size = 0
        for word in self.word_stat:
            size += self.word_s(word)
        return size


class Collocation:
    def __init__(self, w1, w2, p1, p2, wh, sen):
        self.word1 = w1
        self.pos1 = str(p1)
        self.word2 = w2
        self.pos2 = str(p2)
        self.number = Easy.zeroes(spector * 2 + 1)
        self.number[wh] = 1
        # self.total_number = 1
        # if (math.fabs(wh - spector) == 1):
        #     self.near_number = 1
        # else:
        #     self.near_number = 0
        self.sentences = []
        for i in range(spector * 2 + 1):
            self.sentences.append([])
        # print(self.sentences)

        # self.sentences_near = []
        # if (math.fabs(wh - spector) == 1):
        #     self.sentences_near.append(sen)
        # else:
        #     self.sentences.append(sen)
        self.sentences[wh].append(sen)

    def is_equal(self, w1, w2, p1, p2):
        if (w1 == self.word1) & (w2 == self.word2) & (p1 == self.pos1) & (p2 == self.pos2):
            return 1
        return 0

    def add(self, wh, sen):
        self.number[wh] += 1
        # self.total_number += 1
        # if (math.fabs(wh - spector) == 1):
        #     self.near_number += 1
        # self.sentences.append(sen)
        # if (math.fabs(wh - spector) == 1):
        #     self.sentences_near.append(sen)
        # else:
        #     self.sentences.append(sen)
        self.sentences[wh].append(sen)

    def total_number(self):
        sum = 0
        i = 0
        while i < len(self.number):
            sum += self.number[i]
            i += 1
        return sum

    def total_near(self):
        return self.number[spector - 1] + self.number[spector + 1]

    def total_documents(self):
        in_doc = []
        for sen in self.sentences[1]:
            if sen[0] not in in_doc:
                in_doc.append(sen[0])
        for sen in self.sentences[3]:
            if sen[0] not in in_doc:
                in_doc.append(sen[0])
        return len(in_doc)

    def in_doc_number(self):
        in_doc = {}
        for sen in self.sentences[1]:
            if sen[0] not in in_doc:
                in_doc[sen[0]] = 1
            else:
                in_doc[sen[0]] += 1
        for sen in self.sentences[3]:
            if sen[0] not in in_doc:
                in_doc[sen[0]] = 1
            else:
                in_doc[sen[0]] += 1
        return in_doc

    def __str__(self):
        return "%s --- %s : %s  total number = %d  near = %d" % (
            self.word1, self.word2, Easy.to_str(self.number), self.total_number(), self.total_near())

    def is_word(self, w1: str) -> bool:
        if w1 == self.word1:
            return 1
        return 0

    def measures(self, w1, w2, n, d):
        tn = self.total_near()
        td = self.total_documents()
        col_in_doc = self.in_doc_number()

        # print("%d  %d  %d  %d" % (w1, w2, tn, n))

        dice = Stat_func.dice(w1, w2, tn)
        logdice = Stat_func.logdice(w1, w2, tn)
        mi = Stat_func.mi(w1, w2, tn, n)
        mi3 = Stat_func.mi3(w1, w2, tn, n)
        mi_log = Stat_func.mi_log(w1, w2, tn, n)
        t_score = Stat_func.t_score(w1, w2, tn, n)
        tf_idf = Stat_func.tf_idf(tn, td, d)
        tv = Stat_func.tv(tn, col_in_doc, d)

        return "%s; %s; %s; %d; %f; %f; %f; %f; %f; %f; %f; %f" % (
            self.word1, self.word2, Easy.to_str(self.number), tn, dice, logdice, mi, mi3, mi_log, t_score, tf_idf, tv)


def find_big_colloction(st):
    words = st.split()
    total_numb = 0
    left_col = {}
    right_col = {}

    col = []
    for i in range(len(words)):
        (w1, point) = Easy.clear(words[i])
        if w1 != '':
            if Easy.is_number(w1):
                pm = ['#ЧИСЛО#', 'NUMR']
            else:
                p = morph.parse(w1)[0]
                if p.tag.POS == 'NUMR':
                    pm = ['#ЧИСЛО#', 'NUMR']
                else:
                    pm = [p.normal_form, str(p.tag.POS)]
            col.append(pm)

    files = os.listdir(path="Input/")
    f_num = 0
    print(len(files))
    while f_num < len(files):
        filename = "Input/" + files[f_num]
        with open(filename, 'rb') as f:
            file_list = pickle.load(f)
            i = 0
            while i < len(file_list):
                if file_list[i] == col[0]:
                    k = 1
                    find = 1
                    while k < len(col):
                        if file_list[i + k] != col[k]:
                            find = 0
                            break
                        k += 1
                    if find:
                        total_numb += 1
                        col_l = file_list[i - 1][0] + ' ' + file_list[i - 1][1]
                        col_r = file_list[i + k][0] + ' ' + file_list[i + k][1]
                        if col_l in left_col:
                            left_col[col_l] += 1
                        else:
                            left_col[col_l] = 1
                        if col_r in right_col:
                            right_col[col_r] += 1
                        else:
                            right_col[col_r] = 1
                i += 1
        f_num += 1

    print("Слева от словосочетания встречаются:")
    for key in left_col:
        sp = key.find(' ')
        word = key[0:sp]
        print(word, left_col[key])
    print("Справа от словосочетания встречаются:")
    for key in right_col:
        sp = key.find(' ')
        word = key[0:sp]
        print(word, right_col[key])
