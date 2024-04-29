import mujoco 
import pinocchio as pin
import numpy as np


def setup():
    model = mujoco.load_model_from_path("/home/josyula/Programs/pinochhio_tut/mujoco_menagerie/unitree_go2/scene.xml")
    data = model.data 


    urdf_model_path = "/home/josyula/Programs/pinochhio_tut/unitree_ros/robots/go2_description/urdf/go2_description.urdf"
    mesh_dir = "/home/josyula/Programs/pinochhio_tut/unitree_ros/robots/go2_description/meshes/"
    package_dirs = "/home/josyula/Programs/pinochhio_tut/unitree_ros/robots/go2_description"
    model, collision_model  = pin.buildModelsFromUrdf(urdf_model_path, geometry_types=[pin.GeometryType.COLLISION])
    return model, collision_model

def get_config(model, data): 
    # get initial config. 
    q0 = pin.neutral(model)
    v0 = pin.utils.zero(model.nv)
    a0 = pin.utils.zero(model.nv)


def kinematics(model, data, q, v, a):
    pin.forwardKinematics(model, data, q, v, a)
    pin.computeJointJacobians(model, data, q)
    pin.forwardVelocity(model, data, q, v, a)
    pin.computeCentroidalMomentum(model, data, q, v, a)
    pin.computeCentroidalMomentumTimeVariation(model, data, q, v, a)

def dynamics(model, data, q, v, a):
    pin.computeJointJacobiansTimeVariation(model, data, q, v, a)
    pin.crba(model, data, q, v, a)
    pin.nonLinearEffects(model, data, q, v, a)
    pin.computeCoriolisMatrix(model, data, q, v, a)     
    pin.computeGeneralizedGravity(model, data, q, v, a)
    pin.jointTorques(model, data, q, v, a)

def contact(model, data, q, v, a):
    contact_points = pin.updateGeometryPlacements(model, data, collision_model, q)
    contact_forces = [pin.Force.Zero()]*4
    h = pin.ccrba(model, data, q, v, contact_points, contact_forces)
    hdot = pin.ccrba(model, data, q, v, contact_points, contact_forces, True)



def cost(model, data, q, v, a):
    
    

# compute the centroidal momentum
h = pin.ccrba(model, data, q0, v0, contact_points, contact_forces)

# compute the centroidal momentum rate
hdot = pin.ccrba(model, data, q0, v0, contact_points, contact_forces, True)

