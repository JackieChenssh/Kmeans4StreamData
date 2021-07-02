class K_Means(object):
    def __init__(self, n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 100, tol = 1e-4, random_state = None):
        from sklearn.utils import check_random_state
            
        self.n_clusters = n_clusters # 聚类个数
        self.init = init # 获取聚类中心的方式
        self.max_iter = max_iter # 最大迭代次数
        self.random_state = check_random_state(random_state) # 随机数
        self.tol = tol # 停止聚类的阈值
        self.centers = None
#         self.n_init = n_init # 进行多次聚类，选择最好的一次
        
    # Get Clustering Center by K_Means++ or Random_State
    def center_init(self,X):
        import numpy as np
        from sklearn.utils.extmath import stable_cumsum
        from sklearn.metrics.pairwise import euclidean_distances

        init = self.init
        self.feature = X
        dist_sq = None
        if init == 'k-means++':
            n_clusters,random_state = self.n_clusters,self.random_state
            n_samples, n_features = X.shape
            centers = np.empty((n_clusters, n_features), dtype=X.dtype)

            # 设置随机试验的次数
            n_trials = 2 + int(np.log(n_clusters))

            # 从训练数据中随机选择一个条目作为初始中心
            centers[0] = X[random_state.randint(n_samples)]
            # 初始化最近距离列表
            dist_sq = euclidean_distances(centers[0, np.newaxis], X)
            current_pot = dist_sq.sum()
    
            # 选择剩余中心
            for i in range(1, n_clusters):
                # 抽取候选中心
                rand_vals = random_state.random_sample(n_trials) * dist_sq.sum()
                center_ids = np.clip(np.searchsorted(stable_cumsum(dist_sq), rand_vals), None, dist_sq.size - 1)

                # 计算候选中心到各点的距离,并更新候选中心集
                dist_to_centers = np.minimum(dist_sq, euclidean_distances(X[center_ids], X))
                centers_pot = dist_to_centers.sum(axis = 1)

                # 选取候选中心
                min_index = np.argmin(centers_pot)
                current_pot = centers_pot[min_index]
                dist_sq = dist_to_centers[min_index]
                centers[i] = X[center_ids[min_index]]
        elif init == 'random':
            centers = X[random_state.permutation(n_samples)[:n_clusters]]
            dist_sq = euclidean_distances(centers, X)
        else: 
            raise Exception()

        return centers
    
    ## Predict Using Clustering Center
    def predict(self,X):
        import numpy as np
        from sklearn.metrics.pairwise import euclidean_distances
        return np.argmin(euclidean_distances(X, self.centers), axis = 1)

    def _predict(self,x):
        import numpy as np
        from sklearn.metrics.pairwise import euclidean_distances
        return np.argmin(euclidean_distances(x, self.centers))
    
    # Replace Center by the Center of the Clustering Points
    def fit(self,X):
        import numpy as np
        from sklearn.metrics.pairwise import euclidean_distances
        
        if not self.centers:
            self.centers = self.center_init(X)
        centers = self.centers 
        labels = self.predict(X)
        k = centers.shape[0]
        for iters in range(self.max_iter):
            centers_old = centers.copy()
            for i in range(k):
                centers[i] = X[labels == i].mean(axis = 0)
            if(euclidean_distances(centers, centers_old)[np.arange(k),np.arange(k)].sum() < self.tol):
                break
            labels = self.predict(X)
        return labels