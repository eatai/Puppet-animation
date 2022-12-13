# Puppet-animation: Creating 2D multi-joint puppet animations with Python. 

The demonstrated animation was created as a visualization for to illustrate a fictional/simulated cooperative task.

[Sheybani, Saber, Eduardo J. Izquierdo, and Eatai Roth. "Evolving dyadic strategies for a cooperative physical task." 2020 IEEE Haptics Symposium (HAPTICS). IEEE, 2020.](https://arxiv.org/abs/2004.10558)

[](Media/dyadCartoon.png)

The animation depicts two human subjects performing a cooperative task. Each subject is animated as a planar puppet; individual limb segments are connected in series by revolute joints. The approach could be used similarly to render other planar movement (e.g. planar serial robots, human biomechanics like QWOP game, etc).

### Creating links

Linkages should be saved as .svg with the joint-to-joint vector drawn as a red line and the shape of the linkage in black. Below is the upper arm segment as an svg with the ends of the red line define the shoulder and elbow joints.

[](AgentPuppet/upperArm.svg)

###
To install requirements, run the following from the command line:

'make install'

or individually install the packages in requirements.txt.
