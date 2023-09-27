# Hitting_sim
Dynamical Systems and Controllers used in hitting experiments

## Dependencies
1. numpy
2. scipy
3. pybullet
4. qpsolvers, osqp
5. pclick (only for AGX sim)
6. roboticstoolbox-python (only for AGX sim)

Run `pip install -r requirements.txt`

## Running the controller using Pybullet
Run simulation with `python3 simulation_flux_hit.py`

## Running the controller using AGX
Currently getting inertia matrix from Pybullet

### Get URDF-app
1. Clone the urdf-application in branch fix/model-structure  
```bash
git clone -b fix/model-structure git@git.algoryx.se:algoryx/external/i-am/urdf-application.git
```
2. Follow the docker installation steps here: sudo docker login registry.algoryx.se
3. Log in to the docker registry with the credentials you have to login in gitlab 
```bash
sudo docker login registry.algoryx.se
```

### Get custom AGX scene

1. Navigate to the _Project_ folder
```bash
cd urdf-application/PythonApplication/models/Projects
```
2. Clone the repo containing the custom AGX scene
```bash
git clone https://github.com/Elise-J/iam_sim_agx.git
```

### Start simulation
1. Navigate to the _PythonApplication_ folder of the urdf-application
```bash
cd urdf-application/PythonApplication
```
2. Launch AGX simulation
```bash
sudo python3 ../run-in-docker.py python3 click_application.py --model models/Projects/i_am_project/Scenes/IiwaPybullet.yml --timeStep 0.005 --agxOnly --rcs --portRange 5656 5658 --disableClickSync
```
3. The simulation can be seen at  `http://localhost:5656/`
4. In another terminal, navigate to the root of this repo (*hitting_sim*)
5. Run the controller
```bash
python3 simulation_agx.py
```
