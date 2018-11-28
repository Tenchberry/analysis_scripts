#Analysis script for AMOEBA trajectories - Clustering, RMSD, CV monitoring vs time and contact matricies

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis import align
import MDAnalysis.analysis.encore as encore
import subprocess

import nglview

from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from progressbar import *

from MDAnalysis.analysis.rms import rmsd
import random

def rmsd_vs_time_MDanalysis(u, reference):
    
    #u = mda.Universe(top, traj)
    ref = mda.Universe(reference)

    u.trajectory[0]

    nframes = len(u.trajectory)

    results = np.zeros((nframes, 2 ), dtype=np.float64)
    
    #Performing rmsd calculation for input trajectory file
    
    pos = u.select_atoms("name CA")
    
    widgets = ['Calculating RMSD vs time: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

    pbar = ProgressBar(widgets=widgets, maxval=nframes)
    pbar.start()
    
    for iframe, ts in enumerate(u.trajectory):                             
        
        results[iframe, :] = (u.trajectory.time, rmsd(pos.positions, ref.select_atoms("name CA").positions - ref.select_atoms("name CA").center_of_mass()
                                                  , center=True, superposition=True))
        pbar.update(iframe)
    
    pbar.finish()
    print
    
    time = np.trim_zeros(results[:,0], 'b')
    results_trimmed = np.trim_zeros(results[:,1])

    result = np.column_stack((time,results_trimmed))
    
    #Statistical properties of the RMSD vs time graph obtained
    
    angstrom = u'\u212B'.encode('utf-8')
    
    av_rmsd = np.mean(result[:,1])

    min_rmsd = [ int(result[np.argmin(result[:,1]), 0]) , result[np.argmin(result[:,1]), 1] ]
    max_rmsd = [ int(result[np.argmax(result[:,1]), 0]) , result[np.argmax(result[:,1]), 1] ]


    print(u'''Average RMSD value = %f \u212B'''.encode('utf-8') % av_rmsd)
    print(u'''Minimum RMSD value at %d ps = %f \u212B'''.encode('utf-8')  % (min_rmsd[0], min_rmsd[1]) )  
    print(u'''Maximum RMSD value at %d ps = %f \u212B'''.encode('utf-8') % (max_rmsd[0], max_rmsd[1]) )
    
    return result

class clustering:
    '''Set of clustering algorithms from MDAnalysis and Scikit-Learn for clustering of Trp-cage folding trajectories'''
    
    def __init__(self, top, traj, ncut=25):
        
        self.top = top
        self.traj = traj
        
        univ1 = mda.Universe(self.top, self.traj, verbose=True)
        protein = univ1.select_atoms("protein")
        
        coords_protein = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                           protein).run().results
        univ2 = mda.Merge(protein)            # create the protein-only Universe
        univ2.load_new(coords_protein, format=MemoryReader)
        cut = []

        for i in np.arange(0, len(coords_protein[:, 0, 0]), ncut):
            cut.append(coords_protein[i, :, :])

        coord_cut = np.zeros((len(cut), 304, 3), dtype=np.float64)
        coord_cut[:, :, :] = [cut[i] for i in np.arange(0, len(cut))]
        univ_cut = mda.Universe("folded.pdb", coord_cut)

        self.univ_cut = univ_cut
        self.coords_protein = coords_protein
        self.coord_cut = coord_cut
        self.univ2 = univ2
        self.ncut = ncut
        
    def visualise_structures(self, traj, protein_ref):
        
        '''Visualisation of a set of structures using NGLView:-
            protein_ref - Reference structure used to align the input structures based 
            on rotational and translational matricies'''
        
        ref = mda.Universe(protein_ref)
        
        nframes = len(traj.trajectory)

        alignment = align.AlignTraj(traj, ref, select='protein and name CA', in_memory=True, verbose=True)
        alignment.run()
        
        #Visualisation representation set for Trp-Cage - will expand to general proteins later
        view = nglview.show_mdanalysis(traj)
        view.add_cartoon(selection="protein")
        view.add_licorice('TRP')
        view.add_licorice('PRO')
        view.center(selection='protein', duration=nframes)
        view
    
    def affinity_propagation(self, preference=-6.0):

        '''Performing Affinity Propagation clustering of AMOEBA-run Trp-Cage folding trajectory:-

           Default parameter values from MDAnalysis - damping=0.9, max_iter=500, convergence_iter=50
           Preference reduced to -10 from -1 to reflect local homogenity within the trajectory'''

        print ("Performing Affinity Propagation with input preference = %f" % preference)
        clust = encore.cluster(self.univ_cut, method=encore.AffinityPropagation(preference=preference, verbose=True))
        centroids = [cluster.centroid*self.ncut for cluster in clust]
        ids = [cluster.id for cluster in clust]
        
        print("Clustering complete! - %d clusters formed with average size = %d frames" % (len(ids), np.average([cluster.size for cluster in clust]) ) )
        
        coords_centroids = np.zeros((len(centroids), 304, 3), dtype=np.float64)
        coords_centroids[:, :, :] = [self.coords_protein[centroids[i], :, :] for i in np.arange(0, len(centroids))]
        
        
        protein = self.univ2.select_atoms("protein")
        
        univ_centroids = mda.Merge(protein)
        univ_centroids.load_new(coords_centroids, format=MemoryReader)
        
        ref = mda.Universe("folded.pdb")
        
        nframes = len(univ_centroids.trajectory)
        
        alignment = align.AlignTraj(univ_centroids, ref, select='protein and name CA', in_memory=True, verbose=True)
        alignment.run()
        
        idvscenter = {'Cluster ID' : [ids[i] for i in range(0, len(ids))], 'Centroid Time (ps)' 
                      : [centroids[i]*10 for i in range(0, len(centroids))], 'Cluster Size' : [cluster.size for cluster in clust]}
        
        idtable = pd.DataFrame(data=idvscenter)
        
        
        
        #Visualisation representation set for Trp-Cage - will expand to general proteins later
        view = nglview.show_mdanalysis(univ_centroids)
        view.add_cartoon(selection="protein")
        view.add_licorice('TRP')
        view.add_licorice('PRO')
        view.center(selection='protein', duration=nframes)
        view.player.parameters = dict(frame=True)
        return view, idtable, univ_centroids
    
        
vis, clusts, u_centroids = clustering("trp_ref.pdb", "trpgeom_run1.dcd").affinity_propagation(preference=-8.0)

    
#clusts #Note - get RMSD + AlphaFCV value (need to parse to PLUMED) to be able to plot RMSD vs AlphaFCV plot for cluster
       #centroids
    
res_centroids=rmsd_vs_time_MDanalysis(u_centroids, "folded.pdb")
res_centroids[:,0] = clusts['Centroid Time (ps)']

protein = u_centroids.select_atoms("protein")
with mda.Writer("centroids.xtc", protein.n_atoms) as W:
    for ts in u_centroids.trajectory:
        W.write(protein)

#plot_rmsd_vs_time(res_centroids, "GeomPol - Centroids", (random.random(),random.random(),random.random()), (0, (1, 5)), (0, (1, 5)), ncut=1)
#plot_rmsd_vs_time(res_fin, "1DAlphaFRMSD - GeomPol", (random.random(),random.random(),random.random()), "-", (0, (5, 1)), ncut=10)

#PLUMED call in command line to generate AlphaF and Hbonds CV values for cluster centroids

subprocess.Popen(["plumed driver --plumed alphaF.dat --mf_xtc centroids.xtc"], shell=True)

exfile = open("Clusters", "r")
exlines = exfile.read().splitlines()

Hbondvals = []
alphavals = []
times = []
    
widgets = ['Processing hills file into array: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

pbar = ProgressBar(widgets=widgets, maxval=len(exlines))
pbar.start()

for idx, line in enumerate(exlines):
       if line.split()[1] != "FIELDS" and line.split()[1] != "SET":
            Hbondvals.append(line.split()[2])
            alphavals.append(line.split()[1])
            times.append(line.split()[0])
       pbar.update(idx)
pbar.finish()
print        

AlphaFex = np.zeros((len(alphavals), 3), dtype=np.float64)
AlphaFex[:,0] = times
AlphaFex[:,1] = alphavals
AlphaFex[:,2] = Hbondvals

#Forming array with centroid frame times (in ps), RMSD, and AlphaF and Hbonds CV values

fullarray = np.zeros((len(res_centroids), 4), dtype=np.float64)

fullarray[:,0] = res_centroids[:,0]
fullarray[:,1] = res_centroids[:,1]
fullarray[:,2] = AlphaFex[:,1]
fullarray[:,3] = AlphaFex[:,2]

#%matplotlib auto

fig2 = plt.figure(figsize=(10,10))
ax_cv = fig2.add_subplot(111, projection='3d')
#ax_cv = Axes3D(fig2)

cm = plt.cm.rainbow_r

sc = ax_cv.scatter3D(fullarray[:,3], fullarray[:,2], fullarray[:,1], marker="o", c=fullarray[:,0], s=clusts['Cluster Size']*10, alpha=1.0,
                   vmin=int(np.min(fullarray[:,0])), vmax=int(np.max(fullarray[:,0])),  cmap=cm)


ax_cv.legend(loc="best")
ax_cv.set_xlabel(r"Hbond CV value")
ax_cv.set_ylabel(r"AlphaF CV value")
ax_cv.set_zlabel(r"C$_\alpha$ RMSD ($\AA$)")

plt.colorbar(sc)

labels = ['Centroid %d \n Time=%d ps' % ( i, fullarray[i,0]) for i in range(0, len(fullarray[:,0]))]

#for i in range(0, len(labels)):

x, y, _ = proj3d.proj_transform(fullarray[2,3], fullarray[2,2], fullarray[2,1], ax_cv.get_proj())

label = pylab.annotate(
    labels[2],
    xy=(x, y), xytext=(-20, 20),
    textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

def update_position(e):
    #for i in range(0, len(labels)):
    x2, y2, _ = proj3d.proj_transform(fullarray[2,3],fullarray[2,2],fullarray[2,1], ax_cv.get_proj())
    label.xy = (x2, y2)
    label.update_positions(fig2.canvas.renderer)
    fig2.canvas.draw()

fig2.canvas.mpl_connect('button_release_event', update_position)

pylab.show()

#plt.show()
#fig = plt.figure(figsize=(12,6))
#ax_rmsd = fig.add_subplot(111, projection='3d')
#ax_rmsd.scatter3D(fullarray[:,1], fullarray[:,2], fullarray[:,0], color="red", label="RMSDvsAlphaF - 1DGeomPol")
#ax_rmsd.contour3D(fullarray[:,1], fullarray[:,2], fullarray[:,0], color="red", label="RMSDvsAlphaF - 1DGeomPol")
#ax_rmsd.legend(loc="best")
#ax_rmsd.set_xlabel(r"C$_\alpha$ RMSD ($\AA$)")
#ax_rmsd.set_ylabel(r"AlphaF CV value")
#ax_rmsd.set_zlabel(r"Time (ps)")
#plt.show()    


                   #label="HbondvsAlphaFvsRMSD - 1DGeomPol"
#surf = ax_cv.plot_trisurf(fullarray[:,3], fullarray[:,2], fullarray[:,1], alpha=0.5, antialiased=True,
                   #vmin=int(np.min(fullarray[:,0])), vmax=int(np.max(fullarray[:,0])), cmap=cm)
#wframe = ax_cv.plot_wireframe(fullarray[:,3], fullarray[:,2], FIXTHIS, alpha=0.5,cmap=cm)    
        
    