# %%
''' Findmatch.py '''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial import distance

# %%
def matches_coord(matches,kp_left,kp_right):
    m=[]
    for i,j in matches:
        coord = np.zeros(4)
        coord[0], coord[1] = kp_left[i].pt
        coord[2], coord[3] = kp_right[j].pt
        # print(coord)
        m.append(coord)
    return m

# %%
def constructA(p):
    A=[]
    for xx,xy, yx,yy in p:
        A.append([-yx, -yy, -1, 0,0,0, xx*yx, xx*yy, xx])
        A.append([0,0,0, -yx, -yy, -1, xy*yx, xy*yy, xy])
    return np.array(A)

def calculate_res(h, m):
    xs = np.array(m)[:, 0:2]
    yt = np.zeros((len(m), 2))
    i=0
    for xx,xy, yx,yy in m:
        tmp = np.array([yx, yy, 1]).T
        mul = np.dot(h,tmp)
        yt[i] = mul[0:2]/mul[2]
        i=i+1
    res = np.sum((xs-yt)**2, axis=1)
    # print(xs[0], yt[0], res[0])
    return res

def find_inliers(residual, threshold):
    avg = 0
    idx=[]
    res = []
    for i, r in enumerate(residual):
        if r<=threshold:
            idx.append(i)
            res.append(r)
    if(len(res)!=0): 
        avg = sum(res)/len(res)
    return idx, avg

# %%
def Ransac(matches):
    import random
    # RANSAC
    iteration = 2000
    threshold = 2
    max_inliers=0
    H=np.zeros((3,3))
    inliers=[]
    out_avg_res=0

    # for each iteration
    for _ in range(iteration):
        # randomly select four matches
        p = random.sample(matches, 4)
        # print(p)

        # find a model for these matches
        # Ah= 0
        A = constructA(p)
        U,s,V = np.linalg.svd(A)
        h = V[len(V)-1].reshape(3, 3)

        # calculate the distance of other matches with this model
        residual = calculate_res(h, matches)

        # select close enough matches as inliers and reject the others
        inliers_idx, avg_res = find_inliers(residual, threshold)

        # select the model that have the most inliers
        if len(inliers_idx) > max_inliers:
            max_inliers = len(inliers_idx)
            H=h.copy()
            out_avg_res = avg_res
            inliers = [matches[i] for i in inliers_idx]

    print(len(inliers), out_avg_res)
    # fig, ax = plt.subplots(figsize=(20,10))
    # plot_inlier_matches(ax, _gray_left, _gray_right, np.array(inliers))
    # plt.savefig("inliers_matches.jpg")
    return inliers, out_avg_res

# %%
def find_matches(leftpath, rightpath,th_des = 15,threshold = 2):
    img_left = cv.imread(leftpath)
    img_right = cv.imread(rightpath)

    sift = cv.SIFT_create()
    _gray_left = cv.cvtColor(img_left,cv.COLOR_BGR2GRAY)
    # des: (Number of Keypoints)×128.
    kp_left, des_left = sift.detectAndCompute(_gray_left,None)
    # print(des_left[0])
    img_kp_left = cv.drawKeypoints(_gray_left, kp_left, img_left, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_kp_left)

    # sift = cv.SIFT_create()
    _gray_right = cv.cvtColor(img_right,cv.COLOR_BGR2GRAY)
    kp_right, des_right = sift.detectAndCompute(_gray_right,None)
    img_kp_right = cv.drawKeypoints(_gray_right, kp_right, img_right, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure()
    plt.imshow(img_kp_right)

    cv.imwrite('img_kp_right.jpg',img_kp_right)
    des_distance = distance.cdist(des_left,des_right,'sqeuclidean')
    print("des_distance min:", des_distance.min())
    print("des_distance shape:",des_distance.shape)
    matches = []
    th = des_distance.min()*th_des
    for i in range (des_distance.shape[0]):
        for j in range (des_distance.shape[1]):
            if des_distance[i][j] < th:
                matches.append((i,j))
    print(len(matches))
    # matches = np.array(matches)
    inliers, out_avg_res = Ransac(matches_coord(matches,kp_left,kp_right))
    return np.array(inliers), out_avg_res

# %%

# inliers, out_avg_res = find_matches("./data_p1/{}.jpg".format("left"), "./data_p1/{}.jpg".format("right"))
# print(len(inliers), out_avg_res)
# print(inliers)


