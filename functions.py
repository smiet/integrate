# functions.py: supplies several functions for integration, including the dipole,
# the Hopf fibration field, and their isotrope fields
#
#
#
#

import numpy as np


def Dipole(xx, m = np.array([0,0,1])):
    """
    Returns the vector field of a dipole in arbitrary direction
    """
    xxn = xx/np.sqrt(np.sum(xx**2)) #xxnormalized Careful around Zero!
    vf = (3*xxn * np.dot(xxn, m) - m)/np.sqrt(np.sum(xx**2))**3 # Vector Field!
    return vf


def Dipole_guide(xx, m = np.array([0,0,1]), d=np.array([0,0,-.5]),strength = 0.5):
    """
    Returns the vector field of a dipole in arbirary direction plus a guide field in another
    arbitrary direction.
    Args:
    *xx**:
        position where the field is to be evaluated.
    **m**
        Direction of the dipole.
    **d**:
        The constant guide field.
    """
    xxn = xx/np.sqrt(np.sum(xx**2)) #xxnormalized Careful around Zero!
    vf = np.array((3*xxn * np.dot(xxn, m) - m)/np.sum(xxn)**3) # dipole vector field
    return np.array(vf+d*strength)

def Dipole_isotropes(xx):
    """
    Returns the isotropes.
    Note: corresponds with the dipole where m=(0,0,1)!!!
    """
    isotropes= np.zeros(3)

    isotropes[0] = (-9*xx[0]*xx[2]*(xx[0]**2 + xx[1]**2 + 2*xx[2]**2))/((xx[0]**2 + xx[1]**2 + xx[2]**2)*(xx[0]**2 + xx[1]**2 + 4*xx[2]**2))**1.5
    isotropes[1] = (-9*xx[1]*xx[2]*(xx[0]**2 + xx[1]**2 + 2*xx[2]**2))/((xx[0]**2 + xx[1]**2 + xx[2]**2)*(xx[0]**2 + xx[1]**2 + 4*xx[2]**2))**1.5
    isotropes[2] = (-9*xx[2]**2*(xx[0]**2 + xx[1]**2 + 2*xx[2]**2))/((xx[0]**2 + xx[1]**2 + xx[2]**2)*(xx[0]**2 + xx[1]**2 + 4*xx[2]**2))**1.5
    return isotropes

def Dipole_nulls(strength, direction):
    """
    Returns the positions of the null points in the dipole field in a constant guide
    field of magnitude given by strength and direction given by the (cartesian) vector direction.

    Expressions explained in paper/poster.
    """
    nullpos = np.zeros(3)
    R = strength
    #translate the direction vector to shperical coordinates:
    Phi = np.arctan2(np.sqrt(direction[0]**2+direction[1]**2), -direction[2])
    Theta = np.arctan2(-1* direction[1], -1* direction[2])



    nullpos[0] = ((9 + np.cos(2*Phi) + np.sqrt(2)*np.cos(Phi)*np.sqrt(17 +\
        np.cos(2*Phi)))**0.16666666666666666*np.cos(Theta)*np.sin(np.arccos((-1 +\
            np.cos(2*Phi) + np.sqrt(2)*np.cos(Phi)*np.sqrt(17 + np.cos(2*Phi)))/6.)/2.))/\
        (2**0.3333333333333333*R**0.3333333333333333)

    nullpos[1] = ((9 + np.cos(2*Phi) + np.sqrt(2)*np.cos(Phi)*np.sqrt(17 +\
        np.cos(2*Phi)))**0.16666666666666666*np.sin(Theta)* np.sin(np.arccos((-1 +\
            np.cos(2*Phi) + np.sqrt(2)*np.cos(Phi)*np.sqrt(17 +\
                np.cos(2*Phi)))/6.)/2.))/(2**0.3333333333333333*R**0.3333333333333333)

    nullpos[2] = ((9 + np.cos(2*Phi) + np.sqrt(2)*np.cos(Phi)*np.sqrt(17 +\
         np.cos(2*Phi)))**0.16666666666666666*np.cos(np.arccos((-1 + np.cos(2*Phi) +\
             np.sqrt(2)*np.cos(Phi)*np.sqrt(17 +\
                 np.cos(2*Phi)))/6.)/2.))/(2**0.3333333333333333*R**0.3333333333333333)

    return [nullpos, -1*nullpos]



def Hopf_isotropes(xx):
    """
    returns the vector field that is in the direction of the isotropes of the Hopf fibration.
    This means the vectors in the direction that the direction of the Hopf field stays constant.
    """
    x = (4*(xx[1] - xx[0]*xx[2]))/((1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**2*abs(-1 + xx[0]**2 + xx[1]**2 - xx[2]**2))
    y = (-4*(xx[0] + xx[1]*xx[2]))/((1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**2*abs(-1 + xx[0]**2 + xx[1]**2 - xx[2]**2))
    z = (-4*(1 + xx[2]**2))/((1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**2*abs(-1 + xx[0]**2 + xx[1]**2 - xx[2]**2))
    return np.array((x,y,z))


def ZeroField(xx, strength=.1, index=1):
    """
    Returns the magnetic field around a magnetic null. This is equal to the
    linearized field of square vacuum null.

    The null is at the origin and the spine is along the z-axis


    Keyword arguments:

    *xx*
        Position where the field is to be evaulated

    *strength*
        Scales the field strength

    *index*
        Index of the null (is the z-axis in- or out?) only \pm 1 allowed
    """

    factor= {True:-1, False:1}[index>0]
    field=factor*np.array([.5*xx[0], .5*xx[1], -xx[2]])
    return strength*field

def BHopf_guide(xx, strength=0.01, direction=np.array([0,0,1]), **kwargs):
    """
    Function that returns the field of the Hopf field with a guide field in a certain direction

    Keyword arguments:

    *xx*
        Position where the field is to be evaluated

    *strength*
        amplitude of the magnetic field strength

    *direction*
        vector in R3 of unit length that gives the direction of the guide field. If it is not unit length, it is made so.

    other keyword arguments are passed directly on to the Bfield function
    """
    direction= direction/ np.sqrt(np.sum(direction**2))
    hopfField = BHopf(xx, **kwargs)
    return hopfField + (direction*strength)



def BHopf(xx, w1=1, w2=np.sqrt(2), r0=1 ):
        """
        Function that returns the magnetic field of the Hopf map at x,y,z coordinates
        given by xx.

        Functional form to be found in Smiet et al. (2015): 'Self-organizing knotted structures
        in Plasma'

        Call signature:

           BHopf(xx, w1=1, w1=2, r0=1):

       Keyword arguments:

        *xx*
            Position where the field is to be evaluated

        *w1*
            omega1, the poloidal winding number

        *w2*
            omega2, the toroidal winding number

        *r0*
            Scaling factor that increases the size. Equals the radius of the degenerate torus

        """

        bfield=np.zeros(3)
        r2=(np.sum(xx**2)) #
        prefactor = 4*r0**4/(np.pi*(r0**2 + r2)**3)

        bfield[0]=2*(w2*r0*xx[1] - w1*xx[0]*xx[2]) #remember python arrays start at zero!
        bfield[1]=-2*(w2*r0*xx[0] + w1*xx[1]*xx[2])
        bfield[2]= w1*(-1*r0**2 + xx[0]**2 +xx[1]**2 -xx[2]**2)

        return prefactor*bfield
