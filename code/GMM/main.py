import os
from Gaussian_voiceprint_modeling import creat_model
from F_MFCC_E import f_e

'''
闭集声纹辨认：
对每一段音频进行声纹建模，然后对测试音频进行辨认
'''
if __name__ == '__main__':
    root1 = os.path.join("au", "hm.wav")
    GMM_hm = creat_model(root1)  # 根据hm.wav的音频进行声纹建模，hm变为权限者

    root2 = os.path.join("au", "zj.wav")
    GMM_zj = creat_model(root2)

    root3 = os.path.join("au", "zca.wav")
    GMM_zca = creat_model(root3)

    test = os.path.join("au", "hm_test.wav")
    test_feature = f_e(test)
    result = dict(hm=GMM_hm.score(test_feature), zj=GMM_zj.score(test_feature), zca=GMM_zca.score(test_feature))
    print(result)
    result_max = max(result, key=lambda x: result[x])
    print(result_max)
