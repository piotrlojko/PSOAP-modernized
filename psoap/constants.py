import numpy as np
import astropy.constants as aconst
import astropy.units as u


import psoap
import os

PSOAP_dir = os.path.dirname(psoap.__file__)[:-5]

##################################################
# Constants
##################################################
c_ang = aconst.c.to(u.Angstrom / u.s).value #A s^-1
c_kms = aconst.c.to(u.km / u.s).value #km s^-1

#n @ 3000: 1.0002915686329712
#n @ 6000: 1.0002769832562917
#n @ 8000: 1.0002750477973053

n_air = 1.000277
c_ang_air = c_ang/n_air
c_kms_air = c_kms/n_air

h = aconst.h.cgs.value #erg s

G = aconst.G.cgs.value #cm3 g-1 s-2
M_sun = aconst.M_sun.cgs.value #g
R_sun = aconst.R_sun.cgs.value #cm
pc = aconst.pc.cgs.value #cm
AU = aconst.au.cgs.value #cm

day = u.day.to(u.s) # [s]
deg = np.pi / 180 # [radians]
km = 1e5 # [cm]

L_sun = aconst.L_sun.cgs.value #erg/s
F_sun = L_sun / (4 * np.pi * R_sun ** 2) #bolometric flux of the Sun measured at the surface

class ChunkError(Exception):
    '''
    Raised when there was a problem evaluating a specific chunk.
    '''
    def __init__(self, msg):
        self.msg = msg
