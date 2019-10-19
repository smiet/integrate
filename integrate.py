# Integrate.py: supplies the stream class, which integrates field lines using
# a 6th order Runge-Kutta field line integrator.
#
# the class contains built-in functions that calculate winding numbers and
# return Poincar\'e sections.
#
#
#
# Author: Christopher Berg Smiet (chris.smiet@gmail.com, csmiet@pppl.gov)
# With much of the code re-used with permission from
# Simon Candelaresi (simon.candelaresi@gmail.com, iomsn1@googlemail.com).
#
#TODO: Add a cartesian Runge-kutta integrator
#TODO: Make the __init__ function apply numba's JIT-compliation if no kwargs are passed through it (Great speedup expected).
#TODO(stretch): Implement other (simplectic) integrators.

import numpy as np
import multiprocessing as mp
from collections.abc import Iterable
from functools import partial # necesarry for stripping functions in multiprocessing
from matplotlib import pyplot as plt


class stream:
    """
    stream -- Holds the traced streamline.
    """

    def __init__(self,xx, vvfn, integration = 'RK6', intdir= 'forward', hMin = 2e-6, hMax = .5, lMax = 10000, tol = 1e-4, iterMax = 100000, **kwargs ):
        """
        Creates, and returns the traced streamline.

        call signature:

          streamInit(vv, p, integration = 'simple', hMin = 2e-3, hMax = 2e4, lMax = 500, tol = 1e-2, iterMax = 1e3, xx = np.array([0,0,0]))

        Trace magnetic streamlines.

        Keyword arguments:

         *vvfn*:
            Function that returns a length 3 array containing the np.array([x,y,z]) components
            of the vector field (is this correct ordering?)

         *integration*:
            Integration method.
            'simple': low order method.
            'RK6': Runge-Kutta 6th order.

        *intdir*:
            'forward': the field is integrated in the direction of the vector field
            'back': the field is integrated in the opposite direction
            'both': the field is integrated in both directions (not implemented)

         *hMin*:
            Minimum step length for and underflow to occur.

         *hMax*:
            Parameter for the initial step length.

         *lMax*:
            Maximum length of the streamline. Integration will stop if l >= lMax.

         *tol*:
            Tolerance for each integration step. Reduces the step length if error >= tol.

         *iterMax*:
            Maximum number of iterations.

         *xx*:
            Initial seed.

        ***kwargs*
            final kwargs are sent to the vvfn
        """
        self.vvfn = vvfn
        if intdir == 'forward':
            vv = vvfn
        elif intdir == 'back':
            def vv(xx, **kwargs): return -1*vvfn(xx, **kwargs)
        self.tracers = np.zeros([iterMax, 3], dtype = 'float32')  # tentative streamline length

        tol2 = tol**2
        dh   = np.sqrt(hMax*hMin) # initial step size

        # declare vectors
        xMid    = np.zeros(3)
        xSingle = np.zeros(3)
        xHalf   = np.zeros(3)
        xDouble = np.zeros(3)

        # initialize the coefficient for the 6th order adaptive time step RK
        a = np.zeros(6); b = np.zeros((6,5)); c = np.zeros(6); cs = np.zeros(6)
        k = np.zeros((6,3))
        a[1] = 0.2; a[2] = 0.3; a[3] = 0.6; a[4] = 1; a[5] = 0.875
        b[1,0] = 0.2;
        b[2,0] = 3/40.; b[2,1] = 9/40.
        b[3,0] = 0.3; b[3,1] = -0.9; b[3,2] = 1.2
        b[4,0] = -11/54.; b[4,1] = 2.5; b[4,2] = -70/27.; b[4,3] = 35/27.
        b[5,0] = 1631/55296.; b[5,1] = 175/512.; b[5,2] = 575/13824.
        b[5,3] = 44275/110592.; b[5,4] = 253/4096.
        c[0] = 37/378.; c[2] = 250/621.; c[3] = 125/594.; c[5] = 512/1771.
        cs[0] = 2825/27648.; cs[2] = 18575/48384.; cs[3] = 13525/55296.
        cs[4] = 277/14336.; cs[5] = 0.25

        # do the streamline tracing
        self.tracers[0,:] = xx
        outside = False
        sl = 0
        l = 0

        if (integration == 'simple'):
            while ((l < lMax) and (sl < iterMax-1) and (not(np.isnan(xx[0]))) and (outside == False)):
                # (a) single step (midpoint method)
                xMid = xx + 0.5*dh*vv(xx, **kwargs)
                xSingle = xx + dh*vv(xMid, **kwargs)

                # (b) two steps with half stepsize
                xMid = xx + 0.25*dh*vv(xx, **kwargs)
                xHalf = xx + 0.5*dh*vv(xMid, **kwargs)
                xMid = xHalf + 0.25*dh*vv(xHalf, **kwargs)
                xDouble = xHalf + 0.5*dh*vv(xMid, **kwargs)

                # (c) check error (difference between methods)
                dist2 = np.sum((xSingle-xDouble)**2)
                if (dist2 > tol2):
                    dh = 0.5*dh
                    if (abs(dh) < hMin):
                        print("Error: stepsize underflow")
                        break
                else:
                    l += np.sqrt(np.sum((xx-xDouble)**2))
                    xx = xDouble.copy()
                    if (abs(dh) < hMin):
                        dh = 2*dh
                    sl += 1
                    self.tracers[sl,:] = xx.copy()
                    if ((dh > hMax) or (np.isnan(dh))):
                        dh = hMax
                    # check if this point lies outside the domain
                    if (np.sqrt(np.sum(xx**2)) > 10) : #not a brilliant cutoff, but it will stop wandering fieldlines. Must implement an 'outside function'
                        outside = True

        if (integration == 'RK6'):
            while ((l < lMax) and (sl < iterMax-1) and (not(np.isnan(xx[0]))) and (outside == False)):
                k[0,:] = dh*vv(xx, **kwargs)
                k[1,:] = dh*vv(xx + b[1,0]*k[0,:], **kwargs)
                k[2,:] = dh*vv(xx + b[2,0]*k[0,:] + b[2,1]*k[1,:], **kwargs)
                k[3,:] = dh*vv(xx + b[3,0]*k[0,:] + b[3,1]*k[1,:] + b[3,2]*k[2,:], **kwargs)
                k[4,:] = dh*vv(xx + b[4,0]*k[0,:] + b[4,1]*k[1,:] + b[4,2]*k[2,:] + b[4,3]*k[3,:], **kwargs)
                k[5,:] = dh*vv(xx + b[5,0]*k[0,:] + b[5,1]*k[1,:] + b[5,2]*k[2,:] + b[5,3]*k[3,:] + b[5,4]*k[4,:], **kwargs)

                xNew  = xx + c[0]*k[0,:]  + c[1]*k[1,:]  + c[2]*k[2,:]  + c[3]*k[3,:]  + c[4]*k[4,:]  + c[5]*k[5,:]
                xNewS = xx + cs[0]*k[0,:] + cs[1]*k[1,:] + cs[2]*k[2,:] + cs[3]*k[3,:] + cs[4]*k[4,:] + cs[5]*k[5,:]

                delta2 = np.dot((xNew-xNewS), (xNew-xNewS))
                delta = np.sqrt(delta2)

                if (delta2 > tol2):
                    dh = dh*(0.9*abs(tol/delta))**0.2
                    if (abs(dh) < hMin):
                        print("Error: step size underflow")
                        break
                else:
                    l += np.sqrt(np.sum((xx-xNew)**2))
                    xx = xNew
                    if (abs(dh) < hMin):
                        dh = 2*dh
                    sl += 1
                    self.tracers[sl,:] = xx
                    if ((dh > hMax) or (np.isnan(dh))):
                        dh = hMax
                    # check if this point lies outside the domain
                    #if ((xx[0] < p.Ox-p.dx) or (xx[0] > p.Ox+p.Lx+p.dx) or (xx[1] < p.Oy-p.dy) or (xx[1] > p.Oy+p.Ly+p.dy) or (xx[2] < p.Oz) or (xx[2] > p.Oz+p.Lz)):
                    if (np.sqrt(np.sum(xx**2)) > 10) : #not a brilliant cutoff, but it will stop wandering fieldlines. Must implement an 'outside function'
                        outside = True
                if ((dh > hMax) or (delta == 0) or (np.isnan(dh))):
                    dh = hMax

        self.tracers = np.resize(self.tracers, (sl, 3))
        self.x = self.tracers[:, 0]
        self.y = self.tracers[:, 1]
        self.z = self.tracers[:, 2]

        self.l = l
        self.sl = sl


    def getCenter(self):
        """
        this function returns the geometrical center of all points in the stream
        line
        """
        center =  np.array([.0,.0,.0])
        center[0]=np.sum(self.tracers[:,0])/np.size(self.tracers[:,0])
        center[1]=np.sum(self.tracers[:,1])/np.size(self.tracers[:,1])
        center[2]=np.sum(self.tracers[:,2])/np.size(self.tracers[:,2])
        return center

    def getRadius(self):
        """
        this function returns the average distance of the points to the
        geometrical center
        """
        center=self.getCenter()
        radius=.0
        radius = np.sum(np.sqrt(np.sum((self.tracers - center)**2 ,axis=1)))/np.size(self.tracers[:,0])
        return radius


    def getNormal(self):
        """
        this function calculates the normal vector, the orientation of the torus
        it does this by calculating the cross product between the difference vector and the
        vector to the point, and averaging these vectors weighedly
        """
        center=self.getCenter()
        differenceVectors = self.tracers[1:] - self.tracers[:-1]  #calculate the vectors from each linepoint to the next
        vectors = np.cross((self.tracers - center)[:-1], differenceVectors) #calculate the cross product between the vector fom the center to the point, and the point to the next
        vectors = vectors/np.sqrt(np.sum((vectors)**2, axis=1))[:,np.newaxis]     #average them to get the normal vector (incorrectly done because all have to be counted equally :/)
        normalsum = np.sum(vectors, axis =0)
        normal = normalsum/np.sqrt(np.sum(normalsum**2))
        return normal

    def getCrossingNr(self, point, normal):
        """
        returns the number of times the streamline crosses the plane defined by the point and the normal vector.
        """
        sides = np.dot( (self.tracers-point), normal)>0 #calculate the side of the plane each point falls on by projecting on the normal vector
        return np.sum((sides[:-1]^sides[1:]))                 #calculate the xor of the shifted sidetables (which is only true if one is shifted) and return the sum


    def getCrossings(self, point, normal):
        """
        returns the indexes in the streamline array where the crossings occur
        """
        sides = np.dot( (self.tracers-point), normal)>0 #calculate the side of the plane each point falls on by projecting on the normal vector
        return np.flatnonzero(sides[:-1]^sides[1:])                 #calculate the xor of the boolean array that is 1 if it is above the plane with itself bitshifted. nonzero elements is where a crossing through the plane has taken place.

    def getPositiveCrossings(self, point, normal):
        """
        returns the indexes in the streamline array where the crossings occur
        """
        sides = np.dot( (self.tracers-point), normal)<0 #calculate the side of the plane each point falls on by projecting on the normal vector
        return np.flatnonzero((sides[:-1]^sides[1:]) & sides[:-1]) #calculate the xor of the boolean array that is 1 if it is above the plane with itself bitshifted. nonzero elements is where a crossing through the plane has taken place. Last and picks only crossings from positive to negative

    def getTwist( self ):
        """
        Returns the winding number of a streamline object, by calculating the orientation of the
        streamline object from the elements of the streamline itself.
        Unstable for large aspect ratio tori, use getTwist_axisymmetric if you know
        the torus lies in the x,y plane
        """
        center = self.getCenter()
        normal = self.getNormal()
        poloidal = self.getCrossingNr(center, normal)  #calculate poloidal winding (times two, as it counts every crossing)
        toroidal = self.getCrossingNr(center, perp(normal)) # calculate toroidal winding (crossings of a plan perpendicular to the normal vector)
        twist = np.nan if toroidal == 0 else float(poloidal)/toroidal
        return twist

    def getTwistAxisymmetric( self ):
        """
        Returns the winding number of a streamline object, assuming it is on a torus in the
        x, y plane.
        """
        center = np.array([0,0,0])
        normal = np.array([0,0,1])
        poloidal = self.getCrossingNr(center, normal)  #calculate poloidal winding (times two, as it counts every crossing)
        toroidal = self.getCrossingNr(center, np.array([0,1,0])) # calculate toroidal winding (crossings of a plan perpendicular to the normal vector)
        twist = np.nan if toroidal == 0 else float(poloidal)/toroidal
        return twist



    def makePoincare(self, point=np.array([0,0,0]), normal =np.array([0,1,0]), verbose = 0):
        """
        returns the points on the plane (defined by point and normal)
        where the streamline crosses
        """
        #calculate two orthonormal vectors in the plane where we are looking for
        #the intersection
        x2=perpZ(normal) #calculate a vector perpendicular to the normal
        x1=np.cross(normal, x2) # and another vector perpendicular to both
        x1/=np.sqrt(np.sum(x1**2))
        if verbose: print( x1, x2)
        crossingIndices=self.getPositiveCrossings(point, normal)
        print(len(crossingIndices))
        crossingsX1X2 = np.empty((len(crossingIndices),2))
        #print(crossingIndices)
        j=0
        for i in crossingIndices:
            crossCoord= pointOnPlane(self.tracers[i], self.tracers[i+1], point, normal)
            crossingsX1X2[j,:]=([np.dot(x1,crossCoord), np.dot(x2,crossCoord)]) #this goes wrong!
            j+=1

        return np.array(crossingsX1X2)

    def makePoincareXZ(self,  verbose = 0):
        """
        returns the Poincare section in the XZ plane.
        """
        #calculate two orthonormal vectors in the plane where we are looking for
        #the intersection
        point=np.array([0,0,0])
        normal =np.array([0,1,0])
        crossingIndices=self.getPositiveCrossings(point, normal)
        print(len(crossingIndices))
        crossingsX1X2 = np.empty((len(crossingIndices),2))
        for j, i in enumerate(crossingIndices):
            crossCoord= pointOnPlane(self.tracers[i], self.tracers[i+1], point, normal)
            crossingsX1X2[j,:]=([crossCoord[0], crossCoord[2]]) #just the x and z coordinates
        return np.array(crossingsX1X2)

    def scalarAlongLine(self, scalar = 'vmag'):
        if scalar == 'vmag':
            return [np.sqrt(np.sum(self.vvfn(self.tracers[i,:])**2)) for i in range(self.sl)]
        if scalar == 'vx':
            return [self.vvfn(self.tracers[i,:])[0] for i in range(self.sl)]
        if scalar == 'vy':
            return [self.vvfn(self.tracers[i,:])[1] for i in range(self.sl)]
        if scalar == 'vz':
            return [self.vvfn(self.tracers[i,:])[2] for i in range(self.sl)]


def pointOnPlane(p1, p2, point, normal):
    """
    calculate the point on the plane that is between the two points.
    """
    if np.sign(dToPlane(p1, point, normal))== np.sign(dToPlane(p2, point, normal)):
        print ('WARNING: POINTS NOT ON DIFFERENT SIDE OF PLANE')
        return
    linevec = p1-p2 #vector along the line
    distance =(np.dot( (point - p1),normal))/(np.dot(linevec, normal)) #see wikipedia, Line-plane_intersection
    return distance*linevec + p1

def perp( vector ):
    """
    returns a vector perpendicular to the input. the vector will lie in the x=y plane.
    """
    perpvector = np.array([1,1,-(vector[0] + vector[1])/vector[2]])
    return  perpvector/np.sqrt(np.sum(perpvector**2))


def perpZ( vector ):
    """
    returns a vector perpendicular to the input vector. If the input vector is in the x,y plane
    this vector is along the z-axis.
    """
    perpvector = np.array([-vector[2]/(vector[0]+vector[1]),-vector[2]/(vector[0]+vector[1]),1])
    return perpvector/np.sqrt(np.sum(perpvector**2))


def dToPlane(dispoint, point, normal):
    return  np.dot( (dispoint-point), normal)



def stream_multi(startingpoints, **kwargs):
    """
    streams multiple streamlines in parallel on as many cpu's as possible.
    args:
    *startingpoints*:
        n by 3 numpy array of x, y, z coordinates
        OR
        Iterable n-tuple of 3-arrays or a generator of 3-arrays
        that indicate the starting points from which the field lines are to be traced.

    **kwargs:
        are all passed to stream
    """

    if isinstance(startingpoints, np.ndarray) and np.shape(startingpoints)[1]==3:
        iterable = list(startingpoints[i,:] for i in range(len(streamline.tracers)))
    elif isinstance(startingpoints, Iterable):
        iterable = startingpoints
    else:
        print('Sorry Dave, I cannot do that (cannot make startingpoints iterable)')
        return

    packedstream = partial(stream, **kwargs)
    pool = mp.Pool()
    streams = pool.map(packedstream, startingpoints, chunksize=1)
    pool.close()
    pool.join()
    return streams

def norm(vector):
    return np.sqrt(np.sum(vector**2))

def rotate_vector(vector, axis, angle):
    """
    returns a the 3d vector rotated over angle 'angle' along axis 'axis'
    """
    rotation_matrix = expm(np.cross(np.eye(3), axis/norm(axis)*angle))
    return np.dot(rotation_matrix, vector)


def circlePoints(normal, radius=1, slide=0, offset=np.array([0,0,0]), npoints =10, rot=0):
    """
    returns an iterable whose elements are numpy arrays with points
    on the edge of a disk with radius and normal.
    args:
    *normal*: orientation of the circle
    *radius*: radius of the circle
    *slide*: scalar to slide center along normal
    *offset*: translate all points by this vector (add)
    *npoints*: number of points in the iterable that is returned.
    *rot*: rotates the points along the circle
    """
    normal = normal/norm(normal)
    p1 = perp(normal)
    p2 = np.cross(normal, p1)
    n=normal*slide
    points = list(np.array([n[0] + radius*(np.cos(t)*p1[0]+ np.sin(t)*p2[0]),
                          n[1] + radius*(np.cos(t)*p1[1]+ np.sin(t)*p2[1]),
                          n[2] + radius*(np.cos(t)*p1[2]+ np.sin(t)*p2[2])])
                            for t in rot+np.linspace(0,2*np.pi, npoints+1)[:-1])
    for i in range(len(points)):
        points[i]=np.array(offset+points[i])
    return points

def linePoints(start, stop, npoints):
    """
    generate a string of points in a line for giving to stream_multi
    """
    v=stop-start
    return list(start+ s*v for s in np.linspace(0,1, npoints))


def Poincareplot_pointset(pointset, xmin =0, xmax =2, ymin = -1, ymax=1):
    """
    Helper figure to easily make the poincare plots visible.
    (Pointset is what is returned by stream.makePoincare)
    """
    fig = plt.figure()
    fig.set_size_inches(3,5)
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    for points in [pointset,]:
        ax.scatter(points[:,0], points[:,1])
    return ax

class zeroPoints:
    """
    Class object that has iterable attributes that are the points to stream from.

    Call signature:
    points= ZeroPoints(loc, spine, sign=1, dist=0.01, npoints=100)

    Class attributes:
    self.forwards:
        the starting points of fieldlines that are to be integrated forwards
    self.backwards:
        the starting points of fieldlines that are to be integrated
    """
    def __init__(self, loc, spine, sign=1, dist=0.05, npoints=100, radius = None):
        """
        Initialize the Zeropoint object.
        """
        if radius == None:
            radius = dist/4
        print('radius is {}'.format(radius))
        fanpoints = circlePoints(spine, radius= radius, offset=loc, slide=dist,
                                  npoints=npoints)
        fanpoints.extend(circlePoints(spine, radius=radius, offset=loc, slide=-dist,
                         npoints=npoints))
        self.fanpoints = fanpoints
        self.spinepoints = [loc + dist*spine, loc-dist*spine]

        if sign:
            self.forward = self.spinepoints
            self.backward = fanpoints
        else:
            self.forward =  fanpoints
            self.backward = self.spinepoints


