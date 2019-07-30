import networkx as nx
import numpy as np
from sklearn import mixture
import sys
import cv2
import matplotlib.pyplot as plt
import math
from networkx.algorithms.flow import shortest_augmenting_path
import glob
from skimage import io, color
import cv2

def getDistribution(data,k=5):

    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    clf.fit(data)
    return clf



def initDistribution(img,bbox):
    foreground=[]
    background=[]
    alpha=[[0 for i in range(img.shape[1])] for j in range(img.shape[0])]
    for i in range(len(img)):
        for j in range(len(img[i])):
            if i>bbox[1] and j>bbox[0] and i<bbox[3] and j<bbox[2]:
                foreground.append([img[i][j][0],img[i][j][1],img[i][j][2]])
                alpha[i][j]=1
            else:
                background.append([img[i][j][0],img[i][j][1],img[i][j][2]])

    foreground=np.array(foreground)
    background=np.array(background)
    print("foreground",foreground.shape)
    print("background",background.shape)
    fgmm=getDistribution(foreground)
    bgmm=getDistribution(background)
    return fgmm,bgmm,np.array(alpha)

def updateDistribution(img,bbox,alpha):
    foreground=[]
    background=[]
    for i in range(len(img)):
        for j in range(len(img[i])):
            if alpha[i][j]==0:
                foreground.append([img[i][j][0],img[i][j][1],img[i][j][2]])
            else:
                background.append([img[i][j][0],img[i][j][1],img[i][j][2]])

    foreground=np.array(foreground)
    background=np.array(background)
    print("foreground",foreground.shape)
    print("background",background.shape)
    fgmm=getDistribution(foreground)
    bgmm=getDistribution(background)
    return fgmm,bgmm

def createMap(img):
    mapper={}
    invmapper={}
    invmapper[0]='s'
    invmapper[1]='t'
    mapper['s']=0
    mapper['t']=-1
    counter=1
    for i in range(len(img)):
        for j in range(len(img[i])):
            mapper[(i,j)]=counter
            invmapper[counter]=(i,j)
            counter+=1
    return mapper,invmapper


def probpixel(gmm,val):
    weights=gmm.weights_
    means=gmm.means_
    covs=gmm.covariances_
    val=np.array(val)
    ss=0.0
    k=gmm.predict([val])[0]
    #print(k)
    
    det=np.linalg.det(covs[k])
    t1=0.5*math.log(math.sqrt(det)) - math.log(weights[k])
    t2=0.5*np.dot(np.dot(np.transpose(val-means[k]),np.linalg.inv(covs[k])),(val-means[k]))
    ss=t1+t2
    #print(t2)
    
    return max(ss,0)



def getProbability(gmm,img):
    ma,mi=-100000000000000000000,100000000000000000
    pmask=[[0 for i in range(img.shape[1])] for j in range(img.shape[0])]
    for i in range(len(img)):
        #print(i)
        for j in range(len(img[i])):
            if i>bbox[1] and j>bbox[0] and i<bbox[3] and j<bbox[2]:
                pval=probpixel(gmm,[img[i][j][0],img[i][j][1],img[i][j][2]])
                pmask[i][j]=pval
                if pval<mi:
                    mi=pval
                if pval>ma:
                    ma=pval
    print(ma,mi)
    return pmask    


def getPairWise(val1,val2,beta):
    val1=np.array(val1)
    val2=np.array(val2)
    ssd = sum((val1 - val2)**2)
    #beta=2.0
    return 50*math.exp(-1.0*beta*ssd)

def getNeighbours(i,j,n,m):
    results=[]

    if i<n-1:
        results.append([i+1,j])
    if j<m-1:
        results.append([i,j+1])
    '''
    if i<n-1 and j<m-1:
        results.append([i+1,j+1])
    '''
    return results



def makeGraph(img,fgmask,bgmask):
    K=10000000000000000000000000000000000000000000
    G=nx.Graph()
    G.add_node('s')
    G.add_node('t')
    beta_sum=0.0
    tot=0
    for i in range(len(img)):
        for j in range(len(img[i])):
            G.add_node((i,j))
            neighs=getNeighbours(i,j,len(img),len(img[i]))
            for neigh in neighs:
                v1=np.array([img[i][j][0],img[i][j][1],img[i][j][2]])
                x1,y1=neigh[0],neigh[1]
                v2=np.array([img[x1][y1][0],img[x1][y1][1],img[x1][y1][2]])
                beta_sum+=sum((v1-v2)**2)
                tot+=1
    beta=(beta_sum/tot)*0.5


    for i in range(len(img)):
        #print(i)
        for j in range(len(img[i])):
            if i>bbox[1] and j>bbox[0] and i<bbox[3] and j<bbox[2]:
                G.add_edge('t',(i,j),capacity=fgmask[i][j])
                G.add_edge('s',(i,j),capacity=bgmask[i][j])
            else:
                G.add_edge('s',(i,j),capacity=0)
                G.add_edge('t',(i,j),capacity=K)
            neighs=getNeighbours(i,j,len(img),len(img[i]))
            for neigh in neighs:
                v1=[img[i][j][0],img[i][j][1],img[i][j][2]]
                x1,y1=neigh[0],neigh[1]
                v2=[img[x1][y1][0],img[x1][y1][1],img[x1][y1][2]]
                pw=getPairWise(v1,v2,beta)
                G.add_edge((i,j),(x1,y1),capacity=pw)
    return G







if __name__=="__main__":
    g1=glob.glob('./images/person1*')
    g2=glob.glob('./bboxes/person1*')
    g1.sort()
    g2.sort()
    for niter in range(len(g1)):
        print(g1[niter],g2[niter])
        imgname=g1[niter].split('/')[-1]
        img=cv2.imread(g1[niter])
        #img = color.rgb2lab(img)
        fromCenter = False
        r = cv2.selectROI(img, fromCenter)

        x1,y1,x2,y2=int(r[0]),int(r[1]),int(r[0])+int(r[2]),int(r[1])+int(r[3])
        bbox=[x1,y1,x2,y2]
        bbox=np.loadtxt(g2[niter],delimiter=' ')
        for it in range(10):
            print("Iteration",it)
            if it==0:
                fgmm,bgmm,alpha=initDistribution(img,bbox)
                print(fgmm.weights_)
                print("staring unaries")
                fgmask=getProbability(fgmm,img)
                #plt.imshow(fgmask,cmap='gray')
                #plt.show()
                bgmask=getProbability(bgmm,img)
                print("unaries done")
                
                G=makeGraph(img,fgmask,bgmask)
                print("graph made")
                cut_val,cut_pair=nx.minimum_cut(G,'s','t',flow_func=shortest_augmenting_path)
                p1,p2 = cut_pair
                src=-1
                for i in p1:
                    if i=='s':
                        src=1
                if src==1:
                    for i in p1:
                        if i!='s' and i!='t':
                            alpha[i[0]][i[1]]=0
                    for i in p2:
                        if i!='s' and i!='t':
                            alpha[i[0]][i[1]]=1
                else:
                    for i in p1:
                        if i!='s' and i!='t':
                            alpha[i[0]][i[1]]=1
                    for i in p2:
                        if i!='s' and i!='t':
                            alpha[i[0]][i[1]]=0

                print('cut done')
                print(cut_val)
                print('lens',len(p1),len(p2))
            else:

                try:
                    fgmm,bgmm=updateDistribution(img,bbox,alpha)
                except:
                    fname='./results/'+imgname[:-4]+'.png'
                    cv2.imwrite(fname,255*np.array(alpha,dtype='uint8'))
                    continue
                fgmask=getProbability(fgmm,img)
                bgmask=getProbability(bgmm,img)
                G=makeGraph(img,fgmask,bgmask)

                cut_val,cut_pair=nx.minimum_cut(G,'s','t',flow_func=shortest_augmenting_path)
                p1,p2 = cut_pair
                src=-1
                for i in p1:
                    if i=='s':
                        src=1
                if src==1:
                    for i in p1:
                        if i!='s' and i!='t':
                            alpha[i[0]][i[1]]=0
                    for i in p2:
                        if i!='s' and i!='t':
                            alpha[i[0]][i[1]]=1
                else:
                    for i in p1:
                        if i!='s' and i!='t':
                            alpha[i[0]][i[1]]=1
                    for i in p2:
                        if i!='s' and i!='t':
                            alpha[i[0]][i[1]]=0

        fname='./lab/'+imgname[:-4]+str(it)+'.png'
        #print(fname)
        cv2.imwrite(fname,255*np.array(alpha,dtype='uint8'))
    
