#!/usr/bin/env python
# coding: utf-8

# ### Assignment 1: Registering Prokudin-Gorskii color separations of the Russian Empire
# Due date: Monday, September 11, 11:59:59 PM

# The goal of this assignment is to learn to work with images by taking the digitized Prokudin-Gorskii glass plate images and automaticaimgs_basicy producing a color image with as few visual artifacts as possible. In order to do this, you wiimgs_basic need to extract the three color channel images, place them on top of each other, and align them so that they form a single RGB color image.
# 
# Some details are quite important. You should notice that it matters how you crop the images when you align them -- the separations may not overlap exactly. We have provided an RGB image to check your code on at this location. You should separate this image into three layers (R, G and B), then place each of those layers inside a slightly bigger, aimgs_basic white, layer, at different locations. Now register these three. You can teimgs_basic whether you have the right answer in two ways: first, you shifted the layers with respect to one another, so you know the right shift to register them; second, if you look at the registered picture, the colors should be pure.
# 
# You wiimgs_basic need to implement this assignment in Python, and you should familiarize yourself with libraries for scientific computing and image processing including NumPy and PIL.

# In[325]:


# import data
import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt


# In[326]:


# One such possibility is normalized cross-correlation (NCC), which is simply the dot product between the two images normalized to have zero mean and unit norm
def divide_img(im):
    out=[]
    h = int(im.size[1]/3)
    w = im.size[0]
    pad=int(w*0.1)
    for i in range(3):
        upper=i*h
        box = (0,upper, w,upper + h)
        tmp = im.crop(box)
        img = tmp.crop((pad,pad,tmp.size[0]-pad,tmp.size[1]-pad))
        out.append(img)
    return out

def normalize(m):
    row_sums = m.sum(axis=1, keepdims=True)
    new_matrix = m / row_sums
    # print(new_matrix)
    return new_matrix

def ncc(img1, img2):
    img1_norm=normalize(img1)
    img2_norm=normalize(img2)
    p = np.sum(img1_norm*img2_norm)
    print(p)
    return p

def ssd(img1, img2):
    img1_norm=normalize(img1)
    img2_norm=normalize(img2)
    return np.sum((img1_norm-img2_norm)**2)


# In[327]:


# ssd
import math
def displacement(img1, img2):
    out_i = 0
    out_j = 0
    min_val=math.inf
    max_val = math.inf
    win = 15
    # img1= img1.crop((0,0, img1.size[0], img1.size[1]))
    img1= img1.crop((win, win, img1.size[0]-win, img1.size[1]-win))
    img1_data = np.array(img1)

    for i in range(-win, win):
        for j in range(-win, win):
            # box = (i,j,img2.size[0]+i,img2.size[1]+j)
            box = (win+i,win+j,img2.size[0]-win+i,img2.size[1]-win+j)
            img2_trans = img2.crop(box)
            img2_data = np.array(img2_trans)
            val =ssd(img1_data, img2_data)
            # print(i, j, val)
            if val<min_val:
                min_val = val
                out_i = i
                out_j=j
    return -out_i, -out_j

def merge(im1, im2, x, y, padding=30):
    w = im1.size[0] + padding
    h =im1.size[1] + padding
    im = Image.new('L', (w, h), (255))
    im.paste(im2, (padding//2+x, padding//2+y))

    return im


# In[328]:


# Check the channel
# test=Image.open("check.png")
# plt.figure()
# plt.title("test")
# plt.imshow(test)

# r, g, b = test.split()
# test_out = Image.merge('RGB',(r,g,b))
# plt.figure()
# plt.title("test_out")
# plt.imshow(test_out)
# imgs_basic.append(r)
# imgs_basic.append(g)
# imgs_basic.append(b)


# In[329]:


# name = "00125v" 
# name = "00149v" 
# name = "00153v" # base2
# name = "00351v" 
# name = "00398v" 
name = "01112v" 

im=Image.open("data/{}.jpg".format(name)) 

# im=Image.open("data_hires/{}.tif".format(name)) 

# seperate image into three layers
imgs_basic=divide_img(im)


# In[330]:


import time
start_time = time.time()

out_l=[]
base=0
out_l.append(merge(imgs_basic[base], imgs_basic[base], 0,0))

for i in [1,2]:
    x,y = displacement(imgs_basic[base], imgs_basic[i])
    print(i, "h,w: " ,y,x)
    out_l.append(merge(imgs_basic[base], imgs_basic[i], x,y))

# im_out = Image.merge('RGB', (out_l[0], out_l[2], out_l[1]))
im_out = Image.merge('RGB', (out_l[2], out_l[1], out_l[0]))

end_time = time.time()
total_time = end_time-start_time
print("single-scale time: ", total_time)

title = "{}_{}".format("SSD", name)
plt.figure()
plt.title(title)

plt.imshow(im_out)
im_out.save("output/{}.jpg".format(title))


# #### Multiscale Alignment
# 
# For the high-resolution glass plate scans provided above, exhaustive search over all possible displacements will become prohibitively expensive. To deal with this case, implement a faster search procedure using an image pyramid. An image pyramid represents the image at multiple scales (usually scaled by a factor of 2) and the processing is done sequentially starting from the coarsest scale (smallest image) and going down the pyramid, updating your estimate as you go. It is very easy to implement by adding recursive calls to your original single-scale implementation.

# In[331]:


# load high quality images
# name = "01047u" 
name = "01657u" 
# name = "01861a" 

im=Image.open("data_hires/{}.tif".format(name)) 

# seperate image into three layers
imgs_high=divide_img(im)


# In[332]:


def gaussian_pyramid(img, n=5):
    out = []
    img.mode = 'I'
    img = img.point(lambda i:i*(1./256)).convert('L')
    out.append(img)
    for i in range(n):
        # img = img.filter(ImageFilter.GaussianBlur).resize((img.size[0]//2, img.size[1]//2))
        img = img.resize((img.size[0]//2, img.size[1]//2))
        out.append(img)
    return out

# find displacement
def multi_scale(img1, img2,img_pym_1,img_pym_2):
    x,y = displacement(img_pym_1[n], img_pym_2[n])
    x*=-1
    y*=-1
    win= 15

    for scale in range(n-1, -1, -1):
        img1 = img_pym_1[scale]
        img2 = img_pym_2[scale]

        win = win*2
        img1= img1.crop((win, win, img1.size[0]-win, img1.size[1]-win))
        img1_data = np.array(img1)
        min_val=math.inf
        out_i, out_j=0,0
        
        for i in [2*x, 2*x+1]:
            for j in [2*y, 2*y+1]:
                box = (win+i,win+j,img2.size[0]-win+i,img2.size[1]-win+j)
                img2_trans = img2.crop(box)
                img2_data = np.array(img2_trans)
                val =ssd(img1_data, img2_data)
                # print(i, j, val)
                if val<min_val:
                    min_val = val
                    out_i = i
                    out_j = j

        x, y = out_i, out_j
    return -x,-y

start_time = time.time()

img = imgs_high[0]
n=4

img_pym_r = gaussian_pyramid(imgs_high[0],n)
img_pym_g = gaussian_pyramid(imgs_high[1],n)
img_pym_b = gaussian_pyramid(imgs_high[2],n)
# for i in range(5):
#     plt.figure()
#     plt.imshow(img_pym[i])

img1 = img_pym_b[0]
img2 = img_pym_r[0]
img3 = img_pym_g[0]

fx2, fy2 = multi_scale(img1, img2,img_pym_b,img_pym_r)
fx3, fy3 = multi_scale(img1, img3,img_pym_b,img_pym_g)
print("hB, wB: ",fy2,fx2)
print("hG, wG: ",fy3,fx3)
im_out2 = merge(img1, img2, fx2, fy2, 300)
im_out3 = merge(img1, img3, fx3, fy3, 300)
im_out1 = merge(img1, img1, 0, 0, 300)

im_fin = Image.merge('RGB', (im_out1,im_out3,im_out2))

end_time = time.time()
total_time = end_time-start_time
print("multiscale time: ", total_time)

title = "{}_{}".format("Multiscale", name)

plt.figure()
plt.title(title)
plt.imshow(im_fin)
im_fin.save("output/{}.jpg".format(title))

