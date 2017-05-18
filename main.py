import Cluster as Cluster
import Easy as Easy

from Code import Collocation as Col

stat1 = Col.Statistic()
stat2 = Col.Statistic()

stat1.import_word_stat("1")
stat1.import_collocation("1")

stat2.import_word_stat("2")
stat2.import_collocation("2")

t = 1

print("Size of target collectioin == ", len(stat1.collocations))
print("Size of contrast collectioin == ", len(stat2.collocations))

while t:
    print("Выберите желаемое действие:")
    print("1 - Добавление нового файла в коллекцию")
    print("2 - Загрузка всех файлов из дирректории")
    print("3 - Поиск окружения для заданного слова")
    print("4 - Поиск окружения для словосочетания")
    print("5 - Вывод мер для словосочетания в файл")
    print("6 - Выход.")
    print("7 - Найти объемющие словосочетания для пары")
    print("8 - Вывод мер для слов из файла")
    print("Введите число, обозначающее ваш выбор:")
    choise = input()
    if choise == '1':
        print("Введите название файла:")
        filename = input()
        print("В основную коллекци -- 1, в контрастную -- 2:")
        collection = input()
        if collection == "1":
            if filename.find("/") != -1:
                stat1.file_to_words(filename, 1, "1")
            else:
                stat1.file_to_words(filename, 0, "1")
            print("Файл был добавлен в коллекцию.")
            stat1.save_word_stat("1")
            stat1.save_collocations("1")
        else:
            if collection == "2":
                if filename.find("/") != -1:
                    stat2.file_to_words(filename, 1, "2")
                else:
                    stat2.file_to_words(filename, 0, "2")
                print("Файл был добавлен в коллекцию.")
                stat2.save_word_stat("2")
                stat2.save_collocations("2")
            else:
                print("Ошибка ввода")
    if choise == '2':
        print("Введите названии дирректории:")
        dirname = input()
        print("В основную коллекци -- 1, в контрастную -- 2:")
        collection = input()
        if collection == "1":
            stat1.import_dir(dirname, "1")
            stat1.save_word_stat("1")
            stat1.save_collocations("1")
        else:
            if collection == "2":
                stat2.import_dir(dirname, "2")
                stat2.save_word_stat("2")
                stat2.save_collocations("2")
            else:
                print("Ошибка ввода")

    if choise == '3':
        print("Введите слово:")
        word = input()
        print("В основную коллекци -- 1, в контрастную -- 2:")
        collection = input()
        if collection == 1:
            stat1.find_collocations(word)
        if collection == 2:
            stat2.find_collocations(word)
    if choise == '4':
        print("Введите словосочетание:")
        st = input()
        Col.find_big_colloction(st)
    if choise == '5':
        print("Ввыдеите слово:")
        word = input()
        stat1.col_measures(word, stat2)
    if choise == '6':
        stat1.save_word_stat("1")
        stat1.save_collocations("1")
        stat2.save_word_stat("2")
        stat2.save_collocations("2")
        t = 0
    if choise == '7':
        print("Ввыдеите слово:")
        word = input()
        stat1.tmp(word, stat2)
    if choise == '8':
        print("Введите название файла:")
        filename = input()
        f = open(filename)
        for line in f:
            # [word, _] = Easy.clear(line)
            stat1.col_measures(line, stat2)
    if choise == '9':
        print("Введите название файла:")
        filename = input()
        print("Введите уровень отсечения:")
        level = float(input())
        Cluster.cluster(filename, level)
    if choise == '10':
        print("Введите название файла:")
        filename = input()
        Cluster.custom_cluster(filename)
