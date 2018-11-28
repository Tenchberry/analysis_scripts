import MDAnalysis
from MDAnalysis.tests.datafiles import PSF,DCD,CRD
u = MDAnalysis.Universe(PSF,DCD)
#ref = MDAnalysis.Universe(PSF,DCD)     # reference closed AdK (1AKE) (with the default ref_frame=0)
ref = MDAnalysis.Universe(PSF,CRD)    # reference open AdK (4AKE)

import MDAnalysis.analysis.rms

R = MDAnalysis.analysis.rms.RMSD(u, ref,
                   select="backbone",             # superimpose on whole backbone of the whole protein
                   groupselections=["backbone and (resid 1-29 or resid 60-121 or resid 160-214)",   # CORE
                                                          "backbone and resid 122-159",                                   # LID
                                                                                      "backbone and resid 30-59"],                                    # NMP
                                                                                                 filename="rmsd_all_CORE_LID_NMP.dat")
R.run()

import matplotlib.pyplot as plt
rmsd = R.rmsd.T   # transpose makes it easier for plotting
time = rmsd[1]
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.plot(time, rmsd[2], 'k-',  label="all")
ax.plot(time, rmsd[3], 'k--', label="CORE")
ax.plot(time, rmsd[4], 'r--', label="LID")
ax.plot(time, rmsd[5], 'b--', label="NMP")
ax.legend(loc="best")
ax.set_xlabel("time (ps)")
ax.set_ylabel(r"RMSD ($\AA$)")
fig.savefig("rmsd_all_CORE_LID_NMP_ref1AKE.pdf")
