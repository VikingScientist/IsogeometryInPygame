from nutils import mesh, function, plot
from splipy import *
import splipy.surface_factory as splines
import numpy as np
from scipy import sparse
import time

# initialize matrices and stuff
class Fem:

    def __init__(self, physical, n, p):
        """ __init__(self, physical, n, p)
        Constructor for finite element object
        :param physical : (width, height) of physical domain
        :param n        : (n1,n2), number of elements in computational domain
        :param p        : (p1,p2), polynomial degree of compuational discretization
        """
        # create splipy object
        splinesurf = splines.square() * physical
        splinesurf.reparam((0, physical[0]), (0, physical[1]))
        splinesurf.raise_order(p[0]-1, p[1]-1)
        splinesurf.refine(n[0]-1, n[1]-1)

        # create nutils mesh
        domain, geom = mesh.rectilinear( splinesurf.knots() )
        basis = domain.basis( 'spline', degree = p)

        # pre-compute all system matrices
        derivs = function.outer(basis.grad(geom), basis.grad(geom))
        laplace = derivs[:,:,0] + derivs[:,:,1]
        A = domain.integrate( laplace,                     geometry=geom, ischeme='gauss3')
        M = domain.integrate( function.outer(basis,basis), geometry=geom, ischeme='gauss3')
        # print(M.toarray())

        # pre-compute evaluation matrices
        self.Nu = splinesurf.bases[0].evaluate(np.linspace(splinesurf.start('u'), splinesurf.end('u'),physical[0]))
        self.Nv = splinesurf.bases[1].evaluate(np.linspace(splinesurf.start('v'), splinesurf.end('v'),physical[1]))

        # store all nutils object as class variables
        self.splinesurf = splinesurf
        self.domain     = domain
        self.geom       = geom
        self.basis      = basis
        self.A          = A.toscipy() * 1e2
        self.M          = M.toscipy()
        self.u          = np.matrix(np.zeros((A.shape[0],1)))
        self.b          = np.matrix(np.zeros((A.shape[0],1)))

        self.n          = np.array(n)
        self.p          = np.array(p)
        self.physical   = np.array(physical)
        self.time = {'add_smoke':0, 'get_img':0, 'stop_smoke':0, 'diffuse':0, 'solve':0}

    # manually add smoke by mouse-clicking input
    def add_smoke(self, x0):
        t = time.time()
        # print(x0)
        N = [b.evaluate(p) for b, p in zip(self.splinesurf.bases, x0)]
        w = np.kron(N[0], N[1])
        self.b = 1400 * self.M * w.T
        self.time['add_smoke'] += time.time() - t

    # stop smoke creation by releasing mouse
    def stop_smoke(self):
        t = time.time()
        self.b[:] = 0
        self.time['stop_smoke'] += time.time() - t

    # diffuse existing smoke (physics)
    def diffuse(self, dt):
        t = time.time()
        ### convenience naming, easier to read without all the 'self'
        u = self.u
        M = self.M
        A = self.A
        b = self.b

        ### backwards euler method
        # rhs = M*u + dt*b
        # lhs = M   + dt*A
        # u = sparse.linalg.spsolve(lhs, rhs)

        ### crank-nicolson
        t0 = time.time()
        rhs =(M + dt/2*A)*u + dt*b
        lhs = M + dt  *A
        # u = sparse.linalg.spsolve(lhs, rhs)
        u,_ = sparse.linalg.cg(lhs, rhs, x0=u)
        self.time['solve'] += time.time() - t0

        ### wrap results in scipy array
        self.u = np.matrix(np.reshape(u, (A.shape[0],1)))
        # print(self.u)
        self.time['diffuse'] += time.time() - t


    # get evaluated solution field
    def get_image(self):
        t = time.time()
        cp = np.matrix(np.reshape(self.u, self.splinesurf.shape, order='C'))

        img = self.Nu*cp*self.Nv.T
        img = img.clip(0,255)
        img = img.astype('uint8')
        # print(img[0:10,0:15])
        # print(img)
        # print(type(img))
        # print(img.shape)
        # print(img)
        self.time['get_img'] += time.time() - t
        return img

    def get_time(self):
        result = {i:self.time[i] for i in self.time}
        for i in self.time:
            self.time[i] = 0
        return result

