import pdb
import warnings

class KMEC(object):
    def __init__(self,n_feature = None, max_cluster = 10,nmins = 30,center_init = 'KMeans++',distance = 'Euclidean ',delta = 0.01,tau = 0.1,new_center_num = 2):

        import numpy as np
        
        assert center_init in ('KMeans++','Random'),ValueError()
        assert distance in ('Euclidean '),ValueError()
        assert n_feature is not None,ValueError()

        self.delta = delta                                         # 霍夫丁边界置信度
        self.max_cluster = max_cluster                             # 最大聚类数目
        self.nmins = nmins                                       # 最小批次，用于确定是否需要更新或分裂聚类中心
        self.center_init = center_init                             # 'KMeans++'/'Random' 聚类中心初始化方式
        self.clustered_sample = None                               # 样本点+类别矩阵

        # self.centers = np.empty((max_cluster,n_feature))           # 聚类中心矩阵
        self.centers = np.zeros((max_cluster,n_feature))           # 聚类中心矩阵
        self.centers_num = 0                                       # 聚类中心数目

        self.new_center_num = new_center_num
        self.new_sample = []                                       # 新样本
        self.tau = tau
        
        if distance == 'Euclidean ':
            from sklearn.metrics.pairwise import euclidean_distances
            self.distance = euclidean_distances
        
    def Hoeffding_Bound(self,dis_max,dis_min,sample_count):
        from math import log,sqrt
        return sqrt(-abs(dis_max - dis_min) ** 2  * log(self.delta) / (2 * sample_count))
    
    # Get Clustering Center by K_Means++ or Random_State
    def get_cluster_center(self,X,center_num):
        
        # X without label!
        import random
        import numpy as np

        center_init = self.center_init
        distance = self.distance
        
        if center_init == 'KMeans++':
            n_samples, n_features = X.shape
            centers = np.empty((center_num, n_features), dtype = X.dtype)
            # 设置随机试验的次数
            n_trials = 2 + int(np.log(center_num))
            # 从训练数据中随机选择一个条目作为初始中心
            centers[0] = random.choice(list(X))
            # 初始化最近距离列表
            dist_sq = distance(centers[0, np.newaxis], X)
            current_pot = dist_sq.sum()
            # 选择剩余中心
            for i in range(1, center_num):
                # 抽取候选中心
                centers_trial = random.sample(list(X),n_trials)
                # 计算候选中心到各点的距离,并更新候选中心集
                dist_to_centers = np.minimum(dist_sq, distance(centers_trial, X))
                centers_pot = dist_to_centers.sum(axis = 1)
                # 选取候选中心
                min_index = np.argmin(centers_pot)
                current_pot,dist_sq = centers_pot[min_index],dist_to_centers[min_index][np.newaxis]
                centers[i] = centers_trial[min_index]
        elif init == 'Random':
            centers = np.array(random.sample(list(X),center_num))
            dist_sq = np.min(distance(centers, X),axis = 0)[np.newaxis]
        else: 
            raise Exception()

        return centers,dist_sq
        
    def update(self,X):
        # from tqdm.notebook import tqdm
        from tqdm import tqdm
        for x in tqdm(X):
            self._update(x)
    
    def _update(self,x):
        import numpy as np

        nmins = self.nmins
        new_sample = self.new_sample
        centers = self.centers
        max_cluster = self.max_cluster
        distance = self.distance
        clustered_sample = self.clustered_sample
        
        new_sample.append(x)
        new_sample_num = len(new_sample)

        if new_sample_num >= nmins:
            if not self.centers_num:
                centers[0:2],dist_eq = self.get_cluster_center(np.array(new_sample),2)
                self.centers_num += 2
                clustered_sample = np.hstack((np.array(new_sample),np.zeros((new_sample_num,1)),dist_eq.T))
                clustered_sample[:,-2] = self.predict(clustered_sample[:,:-2])
                self.clustered_sample = clustered_sample
            else:
                # 对新样本进行归类，并将归类信息加入矩阵
                new_sample = np.hstack((np.array(new_sample),self.predict(new_sample)[np.newaxis].T,np.zeros((new_sample_num,1))))
                clustered_sample = np.vstack((clustered_sample,new_sample))
                # 更新所有中心
                centers[:self.centers_num] = np.array([np.mean(clustered_sample[clustered_sample[:,-2] == c,:-2],axis = 0) for c in range(self.centers_num)])             
                clustered_sample[:,-1] = np.min(distance(clustered_sample[:,:-2],centers[:self.centers_num]),axis = 1)

                if self.centers_num < max_cluster:
                    n_sample = clustered_sample.shape[0]
                    # 判断是否需要对聚类效果不好的中心进行分裂
                    # 获得可能需要分裂的点，聚类中距离方差最大的聚类中心
                    dist_var = [np.var(clustered_sample[clustered_sample[:,-2] == c,-1]) for c in range(self.centers_num)]
                    pro_center_id = np.argmax(dist_var)
                    pro_var,pro_dist = dist_var[pro_center_id],clustered_sample[clustered_sample[:,-2] == pro_center_id,:-1]
                    # 需要加入判断是否方差达到阈限的代码

                    # 获得霍夫丁边界，给出结点分割的增益阈限                    
                    # epsilon = max(self.Hoeffding_Bound(np.max(pro_dist),np.min(pro_dist),n_sample),self.tau)
                    epsilon = self.Hoeffding_Bound(np.max(pro_dist),np.min(pro_dist),n_sample)
                    pro_sample = clustered_sample[clustered_sample[:,-2] == pro_center_id,:-2]
                    new_center,_ = self.get_cluster_center(pro_sample,2)
                    
                    dist = distance(np.vstack(pro_sample),new_center)
                    new_distance = np.min(dist,axis = 1)

                    # distance_gain = (pro_var - np.var(new_distance)) * pro_sample.shape[0] / n_sample
                    distance_gain = (pro_var - np.var(new_distance)) * pro_sample.shape[0]
                    
                    if distance_gain > epsilon:
                        new_labels = np.argmin(dist,axis = 1)
                        new_labels_new = new_labels.copy()
                        new_labels_new[new_labels == 0] = pro_center_id
                        new_labels_new[new_labels == 1] = self.centers_num
                        new_labels = new_labels_new
                        centers[pro_center_id] = new_center[0]
                        centers[self.centers_num] = new_center[1]
                        self.centers_num += 1

                        pro_sample[:,-2:] = np.vstack((new_distance,new_labels)).T

            new_sample = []
            
    def predict(self,X):
        import numpy as np
        return np.argmin(self.distance(X, self.centers[:self.centers_num]),axis = 1)

    def _predict(self,x):
        import numpy as np
        return np.argmin(self.distance(x[np.newaxis], self.centers[:self.centers_num]))

if __name__ == '__main__':
    import pandas as pd
    from datetime import datetime
    import numpy as np

    data = pd.read_csv('Live_20210128.csv')
    status_type = list(set(data['status_type']))
    data['status_type'] = pd.Series([status_type.index(s_t) for s_t in data['status_type']])
    data['status_published'] = pd.Series([datetime.strptime(d_t,"%m/%d/%Y %H:%M") for d_t in data['status_published']])
    data['status_published_year'] = pd.Series([d_t.year - 2012 for d_t in data['status_published']])
    data['status_published_month'] = pd.Series([d_t.month for d_t in data['status_published']])
    data['status_published_day'] = pd.Series([d_t.day for d_t in data['status_published']])
    data['status_published_hour'] = pd.Series([d_t.hour for d_t in data['status_published']])
    data['status_published_minute'] = pd.Series([d_t.minute for d_t in data['status_published']])
    del data['status_published'],data['status_id']
    feature_label = list(data.columns)

    feature = np.array(data)
    feature = (feature - np.mean(feature,axis = 0)) / np.std(feature,axis = 0)

    kemc = KMEC(n_feature = len(feature_label))
    kemc.update(feature)
    
    print(kemc.predict(feature))


