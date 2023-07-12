# Hitting_sim
Dynamical Systems and Controllers used in hitting experiments

# Dependencies
1. numpy
2. scipy
3. pybullet
4. qpsolvers, osqp

# AGX Sim
Currently controlling in position and working with Pybullet

## Urdf-application
### Get URDF-app
1. Clone the urdf-application in branch fix/model-structure  
```bash
git clone -b fix/model-structure git@git.algoryx.se:algoryx/external/i-am/urdf-application.git
```
2. Follow the docker installation steps here: sudo docker login registry.algoryx.se
3. Log in to the docker registry with the credentials you have to login in gitlab `sudo docker login registry.algoryx.se`

### Get custom Project scene for IIWA
1. `cd  urdf-application/PythonApplication/models/Projects`
2. Clone this repo: `git clone https://github.com/Elise-J/iam_sim_agx.git`

### Start simulation
1. `cd  urdf-application/PythonApplication`
2. `sudo python3 ../run-in-docker.py python3 click_application.py --model models/Projects/i_am_project/Scenes/IiwaPybullet.yml --timeStep 0.005 --agxOnly --rcs --portRange 5656 5658`
3. go to `http://localhost:5656/` in your browser 

## Pybullet app to move the AGX Sim
1. Clone this repo
2. Install dependencies 
3. Install pClick (https://git.algoryx.se/algoryx/external/click#installing)
4. run `python3 simulation_pybullet_agx.py`