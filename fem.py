from nutils import mesh, function, plot
from splipy import *
import splipy.surface_factory as splines
import numpy as np
from scipy import sparse

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
      self.u          = np.matrix(np.ones((A.shape[0],1)))*128
      self.b          = np.matrix(np.zeros((A.shape[0],1)))
      # self.u[1] = 240
      # self.u[200] = 240

  # manually add smoke by mouse-clicking input
  def add_smoke(self, x0):
      # print(x0)
      size      = 100
      intensity = 1e4
      dist  = 1e16
      min_i = 0
      i     = 0
      for col in range(self.splinesurf.shape[1]):
          for row in range(self.splinesurf.shape[0]):
              cp = self.splinesurf[row,col]
              r2  = np.linalg.norm(cp-x0)**2
              self.b[i] = np.max((0, intensity * (1-r2/size/size)))
              i += 1
      print(np.reshape(self.b, self.splinesurf.shape, order='C'))
      # print(self.b)

      # x,y = self.geom
      # r2  = (x0[0]-x)**2 + (x0[1]-y)**2
      # integrand = self.basis * function.max(intensity*(1-r2/size/size),0)
      # self.b = self.domain.integrate(integrand, geometry=self.geom, ischeme='gauss2')

  # diffuse existing smoke (physics)
  def diffuse(self, dt):
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
      rhs =(M + dt/2*A)*u + dt*b
      lhs = M + dt  *A
      u = sparse.linalg.spsolve(lhs, rhs)

      ### wrap results in scipy array
      self.u = np.matrix(np.reshape(u, (A.shape[0],1)))
      # print(self.u)
      # self.b[:] = 0


  # get evaluated solution field
  def get_image(self):
      cp = np.matrix(np.reshape(self.u, self.splinesurf.shape, order='F'))

      img = self.Nu*cp*self.Nv.T
      img = img.clip(0,255)
      img = img.astype('uint8')
      # print(img[0:10,0:15])
      # print(img)
      # print(type(img))
      # print(img.shape)
      # print(img)
      return img
