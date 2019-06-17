# Integrate
Runge-Kutta integrator with built-in functions for generating Poincare sections.

The Stream class oblect contains information on the streamline. 
The Runge-kutta integration is performed as part of it's __init__ routine. 
this routine takes as first argument a numpy-array of length 3, specifying the (x,y,z)-
coordinates of thes tarting point of the Runge-kutta integration. 
The second argument is a python-function, which takes as input a numpy-array of length 3
and returns a numpy-array of length 3 which is the strength of the vector field
at that location. 

The rest of the key-worded arguments control the type and parameters of the numerical 
Runge-Kutta integration routine. 

Example: 
mystream = stream(np.array(0,1,0), BHopf)


The class stream has attributes: 
.l: the stream line length. 
.sl: the number of points on the streamline
.x: x-coordinates of all the points
.y: y-coordinates of all the points on the streamline
.z: z-coordinates of all the points on the streamline
.tracers: all the coordinates in an (sl, 3) numpy array. 
.vvfn(): The function that was passed this streamline is also an 'attribute' in a sense

Built-in functions: 
getCenter(): returns the geometrical center of the point cloud that is the streamlien
getNormal(): returns a proxy for the normal by averaging (point-center)x(nextpoint-point)
getRadius(): returns the average distance of all points from the geometrical center
getCrossingNr(): returns the number of crossings of the fieldline with a plane. 
getCrossings(): returns the coordinates of the crossings
getPositiveCrossings(): returns only the coordinates where the line passes through into the plane
getTwist(): returns the 'twist' of a field line. Only accurate if the field line actually is on a torus. 
getTwistAxisymmetric(): returns the 'twist' if the field lines lie on tori centered on the origin in the x,y plane. 
makePoincare(): Returns a Poincar\'e section
makePoincareXZ(): Returns a Poincar\'e section on of the XZ plane. 
