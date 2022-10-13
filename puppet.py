from svgpathtools import svg2paths2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as tf


def LawOfCosines(a,b,c):
    #print('a: %0.3f,\tb: %0.3f,\tc: %0.3f,\tA:%0.3f'%(a,b,c,(b**2 + c**2 - a**2)/(2*b*c)))
    return np.arccos((b**2 + c**2 - a**2)/(2*b*c))

def SSStriangle(a,b,c):
    return [LawOfCosines(a,b,c), LawOfCosines(b,c,a), LawOfCosines(c,a,b)]

def svg2Poly(svgPath, scale=1, resPts = 20, centered = True):

    pathObj = svg2paths2(svgPath)
    path = pathObj[0][0]

    T = np.linspace(0, 1, resPts + 1)
    p = np.array([])

    for seg in path:
        for t in T:
            p = np.append(p, seg.point(t))

    px = scale * np.real(p)
    py = scale * np.imag(p)

    if centered:
        px = px - np.mean(px)
        py = py - np.mean(py)

    xy = np.transpose(np.vstack((px, py)))
    return xy

def HRT(ax, pos=[0,0], angle=0):
    return tf.Affine2D().rotate(angle) + tf.Affine2D().translate(pos[0], pos[1]) + ax.transData

def poseAgents(Agents, y, offset):
    for k, Agent in enumerate(Agents):
        thetas = Agent.invKin(np.array([0, y-((-1)**k)*0.8*offset, 1]))
        Agent.poseLimbs(thetas)
        Agent.draw()

class PuppetLimb(object):
    kPts = 20

    def __init__(self, svgPath, scale=1, facecolor=np.array([0.5,0.5,0.5]), edgecolor=None):
        self.pathObj = svg2paths2(svgPath)
        self.svgPath = svgPath

        self.facecolor = facecolor
        self.edgecolor = edgecolor

        self.scale = scale
        #  The call to set scale above automatically runs the below methods
        #         self.makeLinkLine()
        #         self.makePatch()
        self.H = tf.Affine2D().identity()



    # @property
    # def facecolor(self):
    #    return self.__facecolor
    #
    # @facecolor.setter
    # def facecolor(self, facecolor):
    #     self.__facecolor = facecolor
    #     self.patch.set_facecolor = facecolor
    #
    # @property
    # def edgecolor(self):
    #     return self.__edgecolor
    #
    # @facecolor.setter
    # def edgecolor(self, edgecolor):
    #     self.__edgecolor = edgecolor
    #     self.patch.set_edgecolor = edgecolor

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale
        self.makeLinkLine()
        self.makePatch()

    def makeLinkLine(self):
        self.line = self.pathObj[0][-1]
        self.link0 = self.scale * np.array([np.real(self.line.point(0)), np.imag(self.line.point(0)), 0]) + np.array(
            [0, 0, 1])
        self.link1 = self.scale * np.array([np.real(self.line.point(1)), np.imag(self.line.point(1)), 0]) + np.array(
            [0, 0, 1])
        self.joint = self.link1 - self.link0
        self.L = np.linalg.norm(self.joint)

    def makePatch(self, zorder=1):
        self.path = self.pathObj[0][0]

        T = np.linspace(0, 1, self.kPts + 1)
        p = np.array([])

        for seg in self.path:
            for t in T:
                p = np.append(p, seg.point(t))

        self.px = self.scale * np.real(p) - self.link0[0]
        self.py = self.scale * np.imag(p) - self.link0[1]
        xy = np.transpose(np.vstack((self.px, self.py)))
        pa = patches.Polygon(xy, closed=True, facecolor = self.facecolor, edgecolor = self.edgecolor, zorder=zorder)
        self.patch = pa
        return pa

    def transformPatch(self, theta, dp):
        npx = self.px * np.cos(theta) - self.py * np.sin(theta) + dp[0]
        npy = self.px * np.sin(theta) + self.py * np.cos(theta) + dp[1]
        return npx, npy


class Puppet(object):

    def __init__(self, limbsSVGList, thetas=np.zeros(4), scale=1, H0=tf.Affine2D().identity(),
                 facecolor = np.array([0.5,0.5,0.5]), edgecolor=None, zorder=1):
        self.limbs = []
        for limbSVG in limbsSVGList:
            self.limbs.append(PuppetLimb(limbSVG, scale=scale, facecolor=facecolor, edgecolor=edgecolor))

        self.thetas = thetas
        self.anchor = self.limbs[0].link1
        self.H0 = H0
        # calls self.poseLimbs()
        self.scale = scale
        self.zorder = zorder
        self.patches = []
        self.makePatches()
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.ax = None


    @property
    def facecolor(self):
        return self.__facecolor

    @facecolor.setter
    def facecolor(self, facecolor):
        self.__facecolor = facecolor
        # if facecolor == 'limbs':
        #     for k,patch in enumerate(self.patches):
        #         patch.set_facecolor = self.limbs[k].facecolor
        # else:
        for k, patch in enumerate(self.patches):
            patch.set_facecolor = facecolor
            self.limbs[k].patch.set_facecolor = facecolor


    @property
    def edgecolor(self):
        return self.__edgecolor

    @edgecolor.setter
    def edgecolor(self, edgecolor):
        self.__edgecolor = edgecolor
        # if edgecolor == 'limbs':
        #     for k, patch in enumerate(self.patches):
        #         patch.set_edgecolor = self.limbs[k].edgecolor
        # else:
        for k, patch in enumerate(self.patches):
            patch.set_edgecolor = edgecolor
            self.limbs[k].patch.set_edgecolor = edgecolor

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale
        for limb in self.limbs:
            limb.scale = scale

    @property
    def H0(self):
        return self.__H0

    @H0.setter
    def H0(self, H0):
        self.__H0 = H0  # + tf.Affine2D().translate(-self.anchor[0], -self.anchor[1])
        self.poseLimbs()

    def poseLimbs(self, thetas=None):
        if thetas is not None:
            self.thetas = thetas
        else:
            thetas = self.thetas

        lastEndPoint = np.array([0.0, 0.0, 1.0])
        for (limb, theta) in zip(self.limbs, thetas):
            Txy = tf.Affine2D().translate(lastEndPoint[0], lastEndPoint[1])
            Rz = tf.Affine2D().rotate(theta)
            limb.H = Rz + Txy + self.H0
            lastEndPoint += Rz.get_matrix() @ limb.joint
            # print(lastEndPoint)

    def invKin(self, tip):
        # tip should be in homogeneous coordinates
        # solve for shoulder and elbow angles assuming torso is stationary and hand vector stays perpendicular to torso
        # Solve angles for triangle made by shoulder, elbow, and wrist
        wrist = self.H0.inverted().get_matrix() @ tip - np.array([0, self.limbs[3].L, 0]) - np.array([0, 0, 1])
        alpha = np.arctan2(wrist[1] - self.limbs[0].joint[1], wrist[0] - self.limbs[0].joint[0])
        # wrist = tip - 0*np.array([0, self.limbs[3].L, 1])
        # wrist = self.H0.get_matrix()@tip - np.array([0, self.limbs[3].L, 1])
        L = np.linalg.norm(wrist[0:2] - self.limbs[0].joint[0:2])  # length from wrist to shoulder
        phi = SSStriangle(self.limbs[1].L, self.limbs[2].L, L)
        theta0 = alpha - phi[1]
        theta1 = np.pi - phi[2] + theta0
        return [0, theta0, theta1, 0]

    def makePatches(self):
        self.patches = []
        self.poseLimbs()
        for limb in self.limbs:
            self.patches.append(limb.makePatch(zorder = self.zorder))

    def addPatches(self, ax):
        paList = []
        for patch in self.patches:
           paList.append(ax.add_patch(patch))
        self.ax = ax
        self.draw()
        return paList

    def draw(self):
        for (limb, patch) in zip(self.limbs, self.patches):
            patch.set_transform(limb.H + self.ax.transData)
