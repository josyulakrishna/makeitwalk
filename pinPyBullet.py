import pybullet as p 
import pybullet_data 
import pinocchio as pin
import numpy as np
from scipy.optimize import minimize

# urdf_model_path = "/home/josyula/Code/pinochhio_tut/unitree_ros/robots/go2_description/urdf/go2_description.urdf"
urdf_model_path = "/home/josyula/Code/pinochhio_tut/anymal_b_simple_description/urdf/anymal.urdf"

def setup_simulation(enableGUI=True):
    if enableGUI:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    robotId = p.loadURDF(urdf_model_path, [0, 0, 0.5], useFixedBase=False)
    dt = 0.01
    p.setTimeStep(dt)

    revolute_joints = []
    revoulute_joint_names = []
    all_joint_names = []
    for i in range(p.getNumJoints(robotId)):
        joint_info = p.getJointInfo(robotId, i)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        all_joint_names.append(joint_name)
        if joint_type == p.JOINT_REVOLUTE:
            revolute_joints.append(i)
            revoulute_joint_names.append(joint_name)
            p.setJointMotorControl2(robotId, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    
    jointTorques = [0.0 for m in revolute_joints]
    p.setJointMotorControlArray(robotId, revolute_joints, controlMode=p.TORQUE_CONTROL, forces=jointTorques)
    return robotId, planeId, revolute_joints, revoulute_joint_names, all_joint_names

# Function to get the position/velocity of the base and the angular position/velocity of all joints
def getPosVelJoints(robotId, revoluteJointIndices):

    jointStates = p.getJointStates(robotId, revoluteJointIndices)  # State of all joints
    baseState = p.getBasePositionAndOrientation(robotId)  # Position of the free flying base
    baseVel = p.getBaseVelocity(robotId)  # Velocity of the free flying base
    # Reshaping data into q and qdot
    q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(),
                   np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
    qdot = np.vstack((np.array([baseVel[0]]).transpose(), np.array([baseVel[1]]).transpose(),
                      np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).transpose()))

    return q, qdot



def cost(x, q):
    a = x[:model.nq]
    f = x[model.nq:]    
    ref = np.array([3, 3]) - q[:2]
    cost1 = 0.9 * np.linalg.norm(a[:2] - ref)**2 + 0.1 * np.linalg.norm(f)**2
    return cost1

def constraints(x, model, data, q, qdot, joint_names):
    a = x[:model.nv]
    f = x[model.nv:]
    pin.computeAllTerms(model, data, q, np.zeros(model.nv))
    M = data.M
    h = data.nle
    tau = data.tau
    sum = 0
    for j in range(len(joint_names)):
        Jci = pin.computeFrameJacobian(model, data, q, model.getFrameId(joint_names[j]), pin.ReferenceFrame.WORLD)
        sum += Jci.T @ f[j * 6:(j + 1) * 6]
    
    constraint1 = M @ a + h - tau - sum
    return np.linalg.norm(constraint1)

def get_contact_points(robotId, planeId):
    cpoints = p.getContactPoints(robotId, planeId)
    link_indices = [cpoint[3] for cpoint in cpoints]
    return link_indices

if __name__ == "__main__":
    robotId, planeId, revolute_joints, revoulute_joint_names, all_joint_names = setup_simulation(enableGUI=True)
    model = pin.buildModelFromUrdf(urdf_model_path, pin.JointModelFreeFlyer())

    while True:
        contact_points = get_contact_points(robotId, planeId)
        if len(contact_points) == 0:
            p.stepSimulation()
        else: 
            print("Contact points: ", contact_points)
            joint_names = [all_joint_names[i] for i in contact_points]    
            q, qdot = getPosVelJoints(robotId, revolute_joints)
            data = model.createData()

            # Optimization problem setup
            x0 = np.zeros((model.nv + 6 * len(joint_names),))
            opt_result = minimize(lambda x: cost(x, q), x0, constraints=({'type': 'eq', 'fun': lambda x: constraints(x, model, data, q, qdot, joint_names)}), method='SLSQP', options={'maxiter': 100, 'ftol': 1e-3})

            # print("Optimal solution: ", opt_result.x)
            
            # Set torques
            a = opt_result.x[:model.nv]
            f_ext = opt_result.x[model.nv:]
            torques = pin.rnea(model, data, q, qdot, a)
            #apply torques to the joints
            for k,v in enumerate(revolute_joints):
                p.setJointMotorControl2(robotId, v, p.TORQUE_CONTROL, force=torques[k+6])
            p.stepSimulation()




# def cost(x, grad, q):
#     a = x[:model.nq]
#     f = x[model.nq:]    
#     ref  = np.array([3, 3]) - q[:2]
#     cost1 =  0.9*np.linalg.norm(a[:2]-ref)**2 + 0.1*np.linalg.norm(f)**2
#     return cost1

# def constraints(x, grad, model, data, q, v, joint_names):
#     #dynamics constraints 
#     a = x[:model.nv]
#     f = x[model.nv:]
#     #M(q) * a + h(q, v) = tau
#     pin.computeAllTerms(model, data, q, v)
#     M = data.M 
#     h = data.nle
#     tau = data.tau
#     sum = 0
#     for j in range(len(joint_names)):
#         # frame = model.frames[model.getFrameId(j)]
#         Jci = pin.computeFrameJacobian(model, data, q, model.getFrameId(joint_names[j]), pin.pin.LOCAL_WORLD_ALIGNED)
#         sum = sum + Jci.T@f[j*6:(j+1)*6]
    
    
#     constraint1 = M @ a + h - tau - sum
#     constraint1 = np.linalg.norm(constraint1)
#     #make constraint size to be 42 append zeros
#     # constraint1 = np.append(constraint1, np.zeros((6*len(joint_names),)))
#     # print("Constraint: ", constraint1.shape)
#     return constraint1

# def get_contact_points(robotId, planeId):
#     #get id of robot and ground plane in the pybullet simulation
#     cpoints =  p.getContactPoints(robotId, planeId)
#     #get link indices of the robot that are in contact with the ground plane
#     link_indices = []
#     for cpoint in cpoints:
#         link_indices.append(cpoint[3])
#     return link_indices

# if __name__ == "__main__":
#     robotId, planeId, revolute_joints, revoulute_joint_names, all_joint_names = setup_simulation(enableGUI=False)
#     model = pin.buildModelFromUrdf(urdf_model_path, pin.JointModelFreeFlyer())
#     # print("model joint data: ", model.names)
#     q, qdot = getPosVelJoints(robotId, revolute_joints)


#     while True:
#         contact_points = get_contact_points(robotId, planeId)
#         if(len(contact_points) == 0):
#             p.stepSimulation()
#         else: 
#             print("Contact points: ", contact_points)
#             joint_names = []
#             for i in contact_points:
#                 # print("Contact point: ", all_joint_names[i])
#                 joint_names.append(all_joint_names[i])    
#             q, qdot = getPosVelJoints(robotId, revolute_joints)
#             data = model.createData()
#             #model, data, q, v, a, f, joint_names
#             #setup optimization problem
#             opt = nlopt.opt(nlopt.LD_SLSQP, model.nv+6*len(joint_names))
#             opt.set_min_objective(lambda x, grad: cost(x, grad, q))
#             opt.add_equality_constraint(lambda x, grad: constraints(x, grad, model, data, q, qdot, joint_names)) 
#             opt.set_maxtime(10)  # for debug
#             opt.set_xtol_rel(1e-3)  
#             x0 = np.zeros((model.nv+6*len(joint_names),))
#             print("Initial guess: ", x0.shape)
#             x = opt.optimize(x0)
#             print("Optimal solution: ", x)
#             break
#             #set torques
#             # for i in range(len(joint_names)):
#             #     joint_idx = all_joint_names.index(joint_names[i])
#             #     p.setJointMotorControl2(robotId, joint_idx, p.TORQUE_CONTROL, force=x[model.nv+i*6:(i+1)*6])

            
#     p.disconnect()