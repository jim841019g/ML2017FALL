import sys
import numpy as np
from skimage import io
import os


def load_image(datapath):
    files = sorted(os.listdir(datapath),key = lambda x: (len(x),x))
    data = []
    for imgfile in files:
        filename = os.path.join(datapath,imgfile)
        picdata = io.imread(filename)
        data.append(picdata)
    data = np.array(data)
    return data

def eigen(data,datapath,recon_pic):
    data = data.reshape((-1,600*600*3))
    mean = np.mean(data,axis=0)
    size = (600,600,3)
     	
    u,s,v = np.linalg.svd(data-mean,full_matrices = False)
    vt = v.T   
    
 
    recon_filename = os.path.join(datapath,recon_pic)
    re_pic = io.imread(recon_filename)
    re_pic = np.array(re_pic).reshape((-1,600*600*3))
    re_pic = re_pic-mean
    weight = np.dot(re_pic,vt[:,0:4])
    recon = np.dot(weight,v[0:4,:]) + mean 
    recon = recon-np.min(recon)
    recon = recon/np.max(recon)
    recon = (recon*255).astype(np.uint8)
    io.imsave('reconstruction.jpg',recon.reshape(size))

if __name__ == "__main__":
    datapath = sys.argv[1]
    recon_pic = sys.argv[2]
    data = load_image(datapath)
    print('data prepare ok')
    eigen(data,datapath,recon_pic)
    
