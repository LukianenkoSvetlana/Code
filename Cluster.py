# import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import *
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import math
import matplotlib.pyplot as plt
from pylab import *


def custom_cluster(filename):
    f = open(filename)
    source = [row.strip().split(';') for row in f]
    names = [row[0] + " -- " + row[1] for row in source[1:]]
    # data = [map(float(), row[4:]) for row in source[1:]]
    end = 8
    for i in range(end):
        for j in range(1, end - i):
            r_number = 4 + i
            measure = source[0][r_number] + " + " + source[0][r_number + j]
            data = [row[r_number:r_number + j + 1:j] for row in source[1:]]
            # print(data)
            data = norm(data)
            dist = pdist(data, 'euclidean')
            data_linkage = hierarchy.linkage(dist, method='average')
            groups = fcluster(data_linkage, 4, criterion='maxclust')
            mean_val = clust_mean(data, groups)
            f_name = "Measures/" + measure + ".csv"
            save_clust(f_name, names, groups, mean_val, measure)


def cluster(filename, level):
    names, data_for_clust = get_data(filename)

    dist = pdist(data_for_clust, 'euclidean')
    data_linkage = hierarchy.linkage(dist, method='average')
    # hierarchy_draw(data_linkage, names, level)
    # elbow_method(data_for_clust, data_linkage)
    aic_bic(data_for_clust, data_linkage)

    groups = fcluster(data_linkage, 7, criterion='maxclust')

    mean_val = clust_mean(data_for_clust, groups)
    # print(groups)
    save_clust(filename, names, groups, mean_val)


def save_clust(filename, names, groups, mean_val, measure = ""):
    in_dir = filename.rfind('/')
    if in_dir != -1:
        filename = filename[in_dir + 1:]
    out_file = filename + "_clusters.csv"
    fo = open(out_file, 'w')
    if measure == "":
        fo.write("Word1 -- Word2; cluster; Dist\n")
    else:
        print(measure)
        str = "Word1 -- Word2; " + measure + "; Dist\n"
        fo.write(str)

    i = 0
    while i < len(names):
        # print(names[i], "  ", groups[i])
        st = '{}; {}; {}\n'.format(names[i], groups[i], mean_val[i])
        fo.write(st)
        i += 1


def hierarchy_draw(data_linkage, labels, level):
    # """Рисуем дендрограмму и сохраняем её"""
    # plt.figure()
    # hierarchy.dendrogram(z, labels=labels, color_threshold=level, leaf_font_size=5, count_sort=True)
    # plt.show()

    figure()
    dendrogram(data_linkage, labels=labels)
    title(u'Дендрограмма')
    show()


def get_data(filename):
    # """Возвращает списки идентификаторов объектов и матрицу значений"""
    f = open(filename)
    source = [row.strip().split(';') for row in f]
    names = [row[0] + " -- " + row[1] for row in source[1:]]
    # data = [map(float(), row[4:]) for row in source[1:]]
    data = [row[4:] for row in source[1:]]
    print(data)
    return names, norm(data)


def norm(data):
    # """Нормирование данных"""
    # print(data)
    matrix = np.array(data, 'f')
    len_val = len(matrix[1, :])
    for i in range(len_val):
        local_min = matrix[:, i].min()
        if local_min != 0.0:
            matrix[:, i] -= local_min
        local_max = matrix[:, i].max()
        if local_max != 0.0:
            matrix[:, i] /= local_max
    return matrix    #tolist()


def clust_mean(data, groups):
    _data = np.array(data, 'f')
    res = []

    for i in range(0, len(data)):
        clust = groups[i]
        inclust = _data[np.array(groups) == clust]
        mean_val = np.mean(inclust, axis=0)
        res.append(math.sqrt(np.sum((data[i] - mean_val) ** 2)))
    print(len(data), " - ", len(groups), " - ", len(res))
    return res


def wgss(data, groups):
    #    Within groups sum of squares (wgss)
    #    Сумма квадратов расстояний от центроида до каждой точки данных
    #    в многомерном пространстве.
    #    Специально на английском, чтобы объяснить название функции
    _data = np.array(data, 'f')
    res = 0.0
    for clust in groups:
        inclust = _data[np.array(groups) == clust]
        meanval = np.mean(inclust, axis=0)
        res += np.sum((inclust - meanval) ** 2)
    return res


def elbow_method(data_for_clust, data_linkage):
    # -------------- Elbow method (метод локтя) -------------------------
    # print(data_linkage)
    # print(len(data_for_clust))
    l = [1] * len(data_for_clust[:,1])   #l = [1] * len(data_for_clust)
    wgs = wgss(data_for_clust, l)
    elbow = [np.nan, wgs]
    for k in range(2, 10):
        groups = fcluster(data_linkage, k, criterion='maxclust')
        elbow.append(wgss(data_for_clust, groups))

    fig = figure()
    ax = fig.add_subplot('121')  # 2 графика в строке, выбираем первый график
    elbow = np.array(elbow)  # Пусть будет numpy массив, удобней...
    ax.plot(elbow/np.nanmax(elbow), 'o', ls='solid')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1.2])
    ax.set_title(u'Сумма внутригрупповых вариаций')
    ax.set_xlabel(u'Число кластеров')

    ax1 = fig.add_subplot('122')  # выбираем второй график в строке

    ax1.plot((elbow[1]-elbow)/np.nanmax(elbow), 'o', ls='solid')
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 1.2])
    ax1.set_xlabel(u'Число кластеров')
    ax1.set_title(u'Доля объясняемой вариации')

    show()


def aic_bic(data_for_clust, data_linkage):
    # ------------- AIC, BIC ----------------------------------------------
    aics = [np.nan]
    bics = [np.nan]
    for k in range(1, 10):
        g = mixture.GaussianMixture(n_components=k)
        g.fit(data_for_clust)
        aics.append(g.aic(data_for_clust))
        bics.append(g.bic(data_for_clust))

    fig = figure()
    ax = fig.add_subplot('121')
    ax.plot(aics, 'o', ls='solid')
    ax.set_title(u'Информационный критерий Акаике')
    ax.set_xlabel(u'Число кластеров')
    ax1 = fig.add_subplot('122')
    ax1.plot(bics, 'o', ls='solid')
    ax1.set_title(u'Информационный критерий Байеса')
    ax1.set_xlabel(u'Число кластеров')

    # ------------- Проекция кластеров на главные компоненты --------------
    pca = PCA(n_components=2)  # Будем проецировать данные на 2 главные компоненты
    Xt = pca.fit_transform(data_for_clust)
    fig = figure()
    for k in range(2, 8):
        groups = fcluster(data_linkage, k, criterion='maxclust')
        ax = fig.add_subplot(3, 2, k - 1)
        for j, m, c in zip(range(k), 'so><^v', 'rgbmyc'):
            ax.scatter(Xt[groups == (j + 1), 0], Xt[groups == (j + 1), 1], marker=m, s=30, label='%s' % j, facecolors=c)
            ax.set_title('k=%s' % k)
            ax.legend(fontsize=14, loc="lower right")
    # k = 4
    # groups = fcluster(data_linkage, k, criterion='maxclust')
    # ax = fig.add_subplot(3, 2, k - 1)
    # for j, m, c in zip(range(k), 'so><^v', 'rgbmyc'):
    #     ax.scatter(Xt[groups == (j + 1), 0], Xt[groups == (j + 1), 1], marker=m, s=30, label='%s' % j, facecolors=c)
    #     ax.set_title('k=%s' % k)
    #     ax.legend(fontsize=14, loc="lower right")
    fig.suptitle(u'Проекрация кластеров на главные компоненты')

    # ------------- Проекция на главные дискриминантны оси-----------------
    sns.set(color_codes=True)
    rcParams['font.family'] = 'DejaVu Sans'  # Импорт seaborn сбрасывает настройки, устанавливаем их снова
    rcParams['font.size'] = 16
    lda = LinearDiscriminantAnalysis(n_components=2)  # Проецируем данные на 2 главные дискримнационные оси
    fig = figure()
    for k in range(3, 9):
        groups = fcluster(data_linkage, k, criterion='maxclust')
        lda.fit(data_for_clust, groups)
        Xt = lda.transform(data_for_clust)  # Собственно проекция данных
        ax = fig.add_subplot(3, 2, k - 2)
        for j, m, c in zip(range(k), 'so><^v', 'rgbmyc'):
            # Проекции при различном числе кластеров разные (в отличие от главных компонент!)
            # Поэтому и данные выглядят на графиках различно
            ax.scatter(Xt[groups == (j + 1), 0], Xt[groups == (j + 1), 1], marker=m, s=30, label='%s' % j, facecolors=c,
                       zorder=10)
            sns.kdeplot(Xt[:, 0], Xt[:, 1], shade=True, cmap="Blues")
            ax.set_title('k=%s' % k)
            ax.legend(fontsize=14, loc="lower right")
    fig.suptitle(u'Проекрация кластеров на дискриминантные оси')

    show()
