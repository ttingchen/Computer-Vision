# %%
# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
## Fundamental Matrix Estimation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

##
## load images and match files for the first example
##

def display_matches(I1, I2, matches):
    # this is a N x 4 file where the first two numbers of each row
    # are coordinates of corners in the first image and the last two
    # are coordinates of corresponding corners in the second image: 
    # matches(i,1:2) is a point in the first image
    # matches(i,3:4) is a corresponding point in the second image
    matches = matches
    N = len(matches)

    ##
    ## display two images side-by-side with matches
    ## this code is to help you visualize the matches, you don't need
    ## to use it to produce the results for the assignment
    ##

    I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
    I3[:,:I1.size[0],:] = I1
    I3[:,I1.size[0]:,:] = I2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3/256).astype(float))
    ax.plot(matches[:,0],matches[:,1],  '+r')
    ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
    ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
    plt.show()

I1_lib = Image.open('MP4_part2_data/library1.jpg')
I2_lib = Image.open('MP4_part2_data/library2.jpg')
matches_lib = np.loadtxt('MP4_part2_data/library_matches.txt')

I1 = Image.open('MP4_part2_data/lab1.jpg')
I2 = Image.open('MP4_part2_data/lab2.jpg')
matches = np.loadtxt('MP4_part2_data/lab_matches.txt')

N = len(matches)
display_matches(I1, I2, matches)

# %%
def fit_fundamental(matches, normalized = False):
    # matches(i,1:2) is a point in the first image
    # matches(i,3:4) is a corresponding point in the second image

    means = np.mean(matches,axis=0)
    m = matches-means
    msd = np.mean(np.sum(m[:,0:2]**2, axis=1))
    t = np.array([[1, 0, -means[0]],
                [0, 1, -means[1]],
                [0, 0, 1]])

    s = np.array([[1/np.sqrt(msd), 0, 0],
                [0, 1/np.sqrt(msd), 0],
                [0, 0, 1]])

    # Step 3: Create transformation matrix
    T = s @ t
    msd_ = np.mean(np.sum(m[:,2:4]**2, axis=1))
    t_ = np.array([[1, 0, -means[2]],
                    [0, 1, -means[3]],
                    [0, 0, 1]])

    s_ = np.array([[1/np.sqrt(msd_), 0, 0],
                    [0, 1/np.sqrt(msd_), 0],
                    [0, 0, 1]])

    # Step 3: Create transformation matrix
    T_ = s_ @ t_
    if normalized:
        h1 = np.hstack((matches[:, 0:2], np.ones((matches.shape[0],1))))
        m1 = T @ h1.T
        x = m1[0]
        y = m1[1]
        h2 = np.hstack((matches[:, 2:4], np.ones((N,1))))
        m2 = T_ @ h2.T
        x_= m2[0]
        y_= m2[1]
    else:
        x = matches[:, 0]
        y = matches[:, 1]
        x_= matches[:, 2]
        y_= matches[:, 3]
    A = np.array([x*x_, x_*y, x_, y_*x, y_*y, y_, x, y, np.ones(x.shape)]).T
    # print(A)
    U,S,V = np.linalg.svd(A)
    f_init = V[len(V)-1].reshape(3,3)

    U_,S_,V_ = np.linalg.svd(f_init)
    S_[2] = 0
    f = U_@(np.diag(S_))@V_

    if normalized:
        f = T_.T @ f @ T

    return f


# %%
##
## display second image with epipolar lines reprojected 
## from the first image
##

# first, fit fundamental matrix to the matches
normalized = True
def display_epipolar(matches, I2):
    N = len(matches)
    F = fit_fundamental(matches,normalized) # this is a function that you should write
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
    closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())
    print("Normalized:",normalized)
    print("Residual:", np.sum((matches[:,2:4]-closest_pt))/N)

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2).astype(int))
    ax.plot(matches[:,2],matches[:,3],  '+r')
    ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
    ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')
    plt.show()
display_epipolar(matches, I2)

# %%
def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

# %%

## Camera Calibration

def camera_calibration(coord2d, coord3d):
    # Assuming you have 3D points (X, Y, Z) and corresponding 2D points (u, v)
    # Define your 3D points and their corresponding 2D points
    # Let's assume you have N correspondences

    u_homogeneous = np.hstack((coord2d, np.ones((N,1)))).T
    X_homogeneous = np.hstack((coord3d, np.ones((N,1)))).T

    # Construct matrix A
    A = np.zeros((2 * N, 12))
    for i in range(N):
        A[2*i, :4] = -X_homogeneous[:, i]
        A[2*i, 8:] = u_homogeneous[0, i] * X_homogeneous[:, i]
        A[2*i+1, 4:8] = -X_homogeneous[:, i]
        A[2*i+1, 8:] = u_homogeneous[1, i] * X_homogeneous[:, i]

    # Perform SVD on A
    _, _, V = np.linalg.svd(A)

    # Solution is the right singular vector corresponding to the smallest singular value
    P = V[-1, :].reshape((3, 4))

    return P

## Camera Centers
def camera_center(P):
    _, _, V = np.linalg.svd(P)
    C = V[len(V)-1]
    C /= C[3]
    return C


# %%
coord3d = np.loadtxt('MP4_part2_data/lab_3d.txt')

## Camera Calibration
P1 = camera_calibration(matches[:,0:2],coord3d)
print("Camera Projection Matrix 1:")
print(P1)
points_3d_proj, residual = evaluate_points(P1, matches[:,0:2], coord3d)
print("calibration residual:",residual)

P2 = camera_calibration(matches[:,2:4],coord3d)
print("Camera Projection Matrix 2:")
print(P2)
points_3d_proj, residual = evaluate_points(P2, matches[:,2:4], coord3d)
print("calibration residual:",residual)

## Camera Centers
C1= camera_center(P1)
print(C1)
C2= camera_center(P2)
print(C2)

P_library1 = np.loadtxt("MP4_part2_data/library1_camera.txt")
P_library2 = np.loadtxt("MP4_part2_data/library2_camera.txt")

C_library1= camera_center(P_library1)
C_library2= camera_center(P_library2)
print(C_library1)



# %%
from mpl_toolkits.mplot3d import Axes3D

def display_center_triangulate(C1, C2, X, title):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera centers
    ax.scatter(C1[0], C1[1], C1[2], c='r', marker='o', label='Camera 1 Center')
    ax.scatter(C2[0], C2[1], C2[2], c='b', marker='o', label='Camera 2 Center')

    # Plot triangulated points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='g', marker='o', label='Triangulated Points')
    # ax.view_init(50, -90)

    # Set labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title("Camera centers and triangulated 3D points for {}".format(title))
    ax.legend()

    # Show the plot
    plt.show()


# %%
## Triangulation
def triangulation(matches, P1, P2,N=N):
    u_homogeneous = np.hstack((matches[:,0:2], np.ones((N,1)))).T
    u_homogeneous2 = np.hstack((matches[:,2:4], np.ones((N,1)))).T
    Xs=[]
    # Construct matrix A
    for i in range(N):
        A = np.zeros((4, 4))
        A[0] = P1[0] - u_homogeneous[0, i]*P1[2]
        A[1] = P1[1] - u_homogeneous[1, i]*P1[2]
        A[2] = P2[0] - u_homogeneous2[0, i]*P2[2]
        A[3] = P2[1] - u_homogeneous2[1, i]*P2[2]

        # Perform SVD on A
        _, _, V = np.linalg.svd(A)

        # Solution is the right singular vector corresponding to the smallest singular value
        X = V[-1, :].reshape((1,4))
        Xs.append(X/X[:,3])
    Xs =np.array(Xs).reshape((N,4))[:,:3]
    return Xs

Xs_lab = triangulation(matches, P1, P2)
print("Lab:",Xs_lab)

display_center_triangulate(C1, C2, Xs_lab, "Lab")


# %%
Xs_library = triangulation(matches_lib, P_library1, P_library2, len(matches_lib))
print("library:",Xs_library)
display_center_triangulate(C_library1, C_library2, Xs_library, "Library")

# %%
import Findmatch as findmatch

leftpath_house = 'MP4_part2_data/house1.jpg'
rightpath_house = 'MP4_part2_data/house2.jpg'
matches_house, res_house = findmatch.find_matches(leftpath_house, rightpath_house,5)
I1_house = Image.open(leftpath_house)
I2_house = Image.open(rightpath_house)

N = len(matches_house)
display_matches(I1_house, I2_house, matches_house)
display_epipolar(matches_house, I2_house)

# %%
leftpath_gaudi = 'MP4_part2_data/gaudi1.jpg'
rightpath_gaudi = 'MP4_part2_data/gaudi2.jpg'
matches_gaudi, res_gaudi = findmatch.find_matches(leftpath_gaudi, rightpath_gaudi,th_des=6)
I1_gaudi = Image.open(leftpath_gaudi)
I2_gaudi = Image.open(rightpath_gaudi)
N = len(matches_gaudi)
display_matches(I1_gaudi, I2_gaudi, matches_gaudi)
display_epipolar(matches_gaudi, I2_gaudi)


