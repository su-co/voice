from sklearn import mixture
from F_MFCC_E import f_e

'''
sklearn.mixture.GaussianMixture
covariance_type包括{‘full’,‘tied’, ‘diag’, ‘spherical’}四种，分别对应完全协方差矩阵（元素都不为零），
    相同的完全协方差矩阵（HMM会用到），对角协方差矩阵（非对角为零，对角不为零），球面协方差矩阵
max_iter表示使用EM算法进行参数估计的最大迭代次数
n_init表示初始化次数，保留最好的那一次
init_params初始化方式，包括k-均值以及随机化
更多参数信息可参考https://blog.csdn.net/lihou1987/article/details/70833229?utm_source=copy
'''


def creat_model(voiceprint_recording_file):  # 声纹录入，并创建声纹模型
    gmm = mixture.GaussianMixture(  # 创建高斯混合模型
        n_components=4, covariance_type='full',  # 数据较少，所以使用较少的高斯分量，防止过拟合
        max_iter=10, n_init=1, init_params='kmeans'
    )
    features = f_e(voiceprint_recording_file)  # 提取权限者声纹特征
    gmm.fit(X=features)  # 对gmm进行参数估计，相当于用权限者声纹特征进行建模
    return gmm
