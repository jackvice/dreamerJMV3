import numpy as np



def get_center_heatmap_sum(nonzero_count, k) -> float:

    
    x = float(nonzero_count) / 384.0  # normalize to [0,1]
    k = 2.0  # 0.2~close to linear; 0.5~mildly convex; 2.0~convex; 3.0~exponential

    return float(np.expm1(k * x) / np.expm1(k))




print('non zero cout 380, k = 5', get_center_heatmap_sum(380, 5))
print('non zero cout 300, k = 5', get_center_heatmap_sum(300, 5))
print('non zero cout 250, k = 5', get_center_heatmap_sum(250, 5))
print('non zero cout 50, k = 5', get_center_heatmap_sum(50, 5))
print('')
print('non zero cout 380, k = 4', get_center_heatmap_sum(380, 4))
print('non zero cout 300, k = 4', get_center_heatmap_sum(300, 4))
print('non zero cout 250, k = 4', get_center_heatmap_sum(250, 4))
print('non zero cout 50, k = 4', get_center_heatmap_sum(50, 4))
print('')
print('non zero cout 380, k = 3', get_center_heatmap_sum(380, 3))
print('non zero cout 250, k = 3', get_center_heatmap_sum(250, 3))
print('non zero cout 50, k = 3', get_center_heatmap_sum(50, 3))
print('')
print('non zero cout 380, k = 2', get_center_heatmap_sum(380, 2))
print('non zero cout 250, k = 2', get_center_heatmap_sum(250, 2))
print('non zero cout 50, k = 2', get_center_heatmap_sum(50, 2))
print('')
print('non zero cout 380, k = 1', get_center_heatmap_sum(380, 1))
print('non zero cout 250, k = 1', get_center_heatmap_sum(250, 1))
print('non zero cout 50, k = 1', get_center_heatmap_sum(50, 1))
print('')
print('non zero cout 380, k = .5', get_center_heatmap_sum(380, 0.5))
print('non zero cout 250, k = .5', get_center_heatmap_sum(250, 0.5))
print('non zero cout 50, k = .5', get_center_heatmap_sum(50, 0.5))
print('')
print('non zero cout 380, k = .1', get_center_heatmap_sum(380, 0.1))
print('non zero cout 250, k = .1', get_center_heatmap_sum(250, 0.1))
print('non zero cout 50, k = .1', get_center_heatmap_sum(50, 0.1))

      
