import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from puppet import *
import pickle

animation_filename_root = 'dyadCartoon'
saveas_filetype = '.mp4'
darkBGFlag = False

if darkBGFlag:
    animation_filename = animation_filename_root + '_darkBG' + saveas_filetype
else:
    animation_filename = animation_filename_root + '_whiteBG'  + saveas_filetype

USE_REAL_DATA = True
dataPath = "Data/traces.p"
dataScaleFactor = 0.5

# Choose data

duration = 15
fps = 40
dt = 1.0/fps



if USE_REAL_DATA:
    results = pickle.load(open(dataPath, "rb"))
    t, r, y, f1, f2, R1, R2 = results
    r *= dataScaleFactor
    y *= dataScaleFactor

else:
    t = np.arange(0, duration, dt)
    r = np.minimum(t, 1) * np.minimum(duration-t, 1)*(0.3*np.sin(t)+0.2*np.sin(5.7*t) + 0.1*np.sin(7.1*t))
    y = np.minimum(t, 1) * np.minimum(duration-t, 1)*(0.4*np.sin(t+0.2)+0.15*np.sin(5.7*t+0.5) + 0.02*np.sin(7.1*t+3))

    f1 = 5*np.abs(np.sin(t))
    f2 = 5*np.abs(np.cos(t))

    fmin = 0.1
    fmax = 1

    R1 = (f1>2.5)
    R2 = (f2>2.5)
'''
Done
'''

tpad = 1.5
pad = int(tpad*fps)

# Pad some data
t = np.arange(-tpad, duration+tpad, dt)
r = np.concatenate((np.zeros(pad), r, np.zeros(2*pad)))

padEmpty = lambda x: np.concatenate((np.empty(pad), x))
y = padEmpty(y)
f1 = padEmpty(f1)
f2 = padEmpty(f2)
R1 = (padEmpty(R1)).astype(int)
R2 = (padEmpty(R2)).astype(int)

# mpl.rcParams['font.sans-serif'] = ['Myriad Pro']
#BGCOLOR = np.array([38,38,38])/255.0



if darkBGFlag:
    BGCOLOR = np.array([38, 38, 38]) / 255.0
    TABLEEDGECOLOR = 'w'
    TABLECOLOR = np.array([1, 1, 1, 0.25])

else:
    BGCOLOR = np.array([1,1,1,0])
    TABLEEDGECOLOR = np.array([0,0,0])
    TABLECOLOR = np.array([0.3,0.3,0.3, 1])


TCOLOR = np.array([1,1,1])
REFCOLOR = np.array([255,255,255, 255])/255.0
YCOLOR =  np.array([18, 178, 75])/255.0
A1COLOR =  np.array([76, 171, 206])/255.0
A2COLOR =  np.array([255, 203, 65])/255.0

A1COLORb = A1COLOR-0.05*np.array([1,1,1])
A2COLORb = A2COLOR-0.05*np.array([1,1,1])

Role1COLOR =  np.array([226, 31, 38, 255])/255.0
Role2COLOR =  np.array([246, 153, 154, 255])/255.0
clear = np.array([0, 0, 0, 0])

RoleColor = [Role1COLOR, Role2COLOR]
RoleText = [' role T ', ' role S ']

# roleCmap = ListedColormap(np.vstack((Role1COLOR, Role2COLOR)))

params = {"figure.facecolor": BGCOLOR,
          "font.sans-serif": "Verdana",
          "ytick.left": False,
          "ytick.labelleft": False,
          "xtick.bottom": False,
          "xtick.labelbottom": False,
          "axes.facecolor": [0,0,0,0],
          "axes.labelcolor": "w",
          "axes.linewidth": 2,
          "axes.spines.left": False,
          "axes.spines.right": False,
          "axes.spines.top": False,
          "axes.spines.bottom": False,
          "axes.axisbelow": True,
          "axes.labelsize": 16,
          "xtick.labelsize": 16,
          "ytick.labelsize": 16,
          "xtick.major.pad": 9,
          "ytick.major.pad": 6,
          "xtick.major.width": 2,
          "ytick.major.width": 2,
          "xtick.direction": 'in',
          "ytick.direction": 'in'}

plt.rcParams.update(params)


# Scale factor from pixel coordinates
pix2meters = 1/150.0

# Import SVG Files
# torso = puppetLimb('AgentPuppet/torso.svg', scale = pix2meters)
# upperArm = puppetLimb('AgentPuppet/upperArm.svg', scale = pix2meters)
# foreArm = puppetLimb('AgentPuppet/foreArm.svg', scale = pix2meters)
# hand = puppetLimb('AgentPuppet/hand.svg', scale = pix2meters)




zorder = {'table': 0, 'r':0.1, 'egg': 0.5, 'y':1,  'tableEdge': 1.1, 'Role': 1.5, 'Agent':2, 'head': 2.1, 'AgentLabel':2.5}

# Make Table
tableWH = np.array([2*tpad, 2])
table = patches.Rectangle(-0.5*tableWH, tableWH[0], tableWH[1], edgecolor = None, facecolor = TABLECOLOR, zorder=zorder['table'])
tableEdge = patches.Rectangle(-0.5*tableWH, tableWH[0], tableWH[1], edgecolor = TABLEEDGECOLOR, linewidth = 2, facecolor = None, zorder=zorder['tableEdge'])

# Make Egg
eggPts = svg2Poly('AgentPuppet/egg.svg', scale = pix2meters)
eggWH = np.amax(eggPts,0)-np.amin(eggPts,0)
eggHalfHeight = 0.5*eggWH[1]

egg = patches.Polygon(eggPts, facecolor='w', edgecolor = 'k', linewidth = 1, zorder=zorder['egg'])

# Make Agents
thetas = [0, np.pi/4, 3*np.pi/4, 0]

limbsList = ['AgentPuppet/torso.svg',
            'AgentPuppet/upperArm.svg',
            'AgentPuppet/foreArm.svg',
            'AgentPuppet/hand.svg']

Agent=[None]*2
Agent[0] = Puppet(limbsList, scale = pix2meters, thetas=thetas, facecolor=A1COLOR, zorder=zorder['Agent'])
Agent[1] = Puppet(limbsList, scale = pix2meters, thetas=thetas, facecolor=A2COLOR, zorder=zorder['Agent'])

agent_distance = 0.675 * tableWH[1]
Agent[0].H0 = tf.Affine2D().translate(-Agent[0].limbs[0].L, -agent_distance)
Agent[1].H0 = tf.Affine2D().rotate(np.pi) + tf.Affine2D().translate(Agent[1].limbs[0].L, agent_distance)


headPts = svg2Poly('AgentPuppet/head.svg', scale = pix2meters)
Head=[None]*2
Head[0] = patches.Polygon(headPts, facecolor=A1COLORb, edgecolor = None, zorder=zorder['head'])
Head[1] = patches.Polygon(headPts, facecolor=A2COLORb, edgecolor = None, zorder=zorder['head'])

A1head_pos = np.array([-0.35, -agent_distance+0.33])
A2head_pos = -1*A1head_pos

calcAngle = lambda headPos, target: np.arctan2(target[1]-headPos[1], target[0]-headPos[0])


# Make figure
fig, ax = plt.subplots(figsize=(4,4.5))
ax.set_xlim([-tpad-0.1,tpad+0.1])
ax.set_aspect('equal', adjustable='datalim')

a0Patches = Agent[0].addPatches(ax)
a1Patches = Agent[1].addPatches(ax)

ax.add_patch(table)
ax.add_patch(egg)

A1head = ax.add_patch(Head[0])
A2head = ax.add_patch(Head[1])

textPos = [-0.6, -agent_distance + 0.1]
A1_text = ax.text(-0.525, -agent_distance + 0.2, 'A1', fontsize=16, fontweight = 'demi', color = 'w', zorder = zorder['AgentLabel'] )
A2_text = ax.text(0.25, agent_distance - 0.3, 'A2', fontsize=16, fontweight = 'demi', color = 'w', zorder = zorder['AgentLabel'] )

A1_Role = ax.text(0.6, -agent_distance + 0.075, RoleText[0], color='w', fontsize = 12, fontweight = 'bold', zorder = zorder['Role'],
        bbox=dict(facecolor=RoleColor[0], edgecolor='none', boxstyle='round,pad=0.6'))
A2_Role = ax.text(-1.15, agent_distance -0.175, RoleText[0], color='w', fontsize = 12, fontweight = 'bold', zorder = zorder['Role'],
        bbox=dict(facecolor=RoleColor[0], edgecolor='none', boxstyle='round,pad=0.6'))



# Prep for animation, could be init() but prefer not
k = 0
ridx = np.arange(k+2, k + 2 * pad - 1)
idx = np.arange(k+2, k + pad)
rline, = ax.plot(t[ridx], r[ridx], color=REFCOLOR, linewidth=1.5, zorder = zorder['r'], clip_path = table)
yline, = ax.plot(t[idx], y[idx], color = YCOLOR, linewidth=3, zorder = zorder['y'], clip_path = table, solid_capstyle = 'round')

y0=0
egg.set_transform(tf.Affine2D().translate(0,y0)+ax.transData)
poseAgents(Agent, y0, eggHalfHeight)

A1head.set_transform(HRT(ax, A1head_pos, calcAngle(A1head_pos, [0,y0])))
A2head.set_transform(HRT(ax, A2head_pos, calcAngle(A2head_pos, [0,y0])))


plt.tight_layout()



def update(k):

    # k+pad is current time index
    ridx = np.arange(k + 2, k + 2 * pad - 1)
    idx = np.arange(k + 2, k + pad)

    rline.set_ydata(r[ridx])
    yline.set_ydata(y[idx])

    y_curr =  y[k+pad]
    egg.set_transform(tf.Affine2D().translate(0, y_curr)+ax.transData)

    poseAgents(Agent, y[k+pad], eggHalfHeight)

    A1head.set_transform(HRT(ax, A1head_pos, calcAngle(A1head_pos, [0, y_curr])))
    A2head.set_transform(HRT(ax, A2head_pos, calcAngle(A2head_pos, [0, y_curr])))

    A1_text
    A2_text

    if R1[k+pad] is not None:
        A1_Role.set_text(RoleText[R1[k+pad]])
        A1_Role._bbox_patch.set_facecolor(RoleColor[R1[k+pad]])

        A2_Role.set_text(RoleText[R2[k+pad]])
        A2_Role._bbox_patch.set_facecolor(RoleColor[R2[k+pad]])

    artistList = [rline, yline, egg, A1_text, A2_text, A1_Role, A2_Role, A1head, A2head]+a0Patches+a1Patches

    if k%100==1:
        print("Frame %d" %(k-1))
    return artistList

anim = animation.FuncAnimation(fig, func=update, frames=duration*fps, interval = 25, blit=True, repeat=False, )
writer = animation.FFMpegFileWriter(fps=40, bitrate=500)
anim.save(animation_filename, writer=writer, dpi=300, savefig_kwargs={'facecolor': BGCOLOR})

print("Finished")
plt.show()