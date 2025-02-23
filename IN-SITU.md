# In Situ Visualization with Catalyst
If Catalyst is not available as a module on the machine, this is the first to build.
For a more indepth documentation on how to build catalyst check the documentation, here are only minimal steps described.
## Build and Install Catalyst
### Requirements for Catalyst
* MPI

1. Configuration
```sh
git clone https://gitlab.kitware.com/paraview/catalyst.git
cd catalyst
git checkout v2.0.0-rc4
cmake -S . \
	-B build \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCATALYST_USE_MPI=ON \
	-DCMAKE_INSTALL_PREFIX=../catalyst_install
```
2. Build and Test
`-j` flag is for parallel build and parallel test
```sh
cmake --build build -j 4
cmake --install build
ctest --test-dir build/tests -j 4
```
The tests should all pass, however for some of them you need to export the `catalyst_DIR` path.
So inspect the output of the failed ones why they failed!

## Build IPPL with Catalyst
I would recommend to use the install script from IPPL for building IPPL and then after that setting `ENABLE_CATALYST` to `ON`, this procedure would then be:
```sh
./ippl-build-scripts/999-build-everything -t serial -i
cmake ippl/build_serial
```
Now in order you need to pass the `Catalyst_Dir` path to the install script.
You can do that in the cli over:
```sh
cmake build_serial\
  -DENABLE_CATALYST=ON \
  -DCATALYST_DIR=<absolute_path_to_catalyst_install>
```
or using the cmake gui over:
```sh
ccmake build_serial
```

Now build again and you will have catalyst enabled.


## Starting a simulation with Catalyst
An general set for examples can be seen on [https://gitlab.kitware.com/paraview/paraview/-/tree/master/Examples/Catalyst2](https://gitlab.kitware.com/paraview/paraview/-/tree/master/Examples/Catalyst2).
Generally the setup is always similar and starting a simulation is usually as:
```sh
./some-program <program_args> ./<path_to_python_script>
```
Some sample scripts for ippl do lay under the `./test/stream/` directory.
For all scripts the `CATALYST_IMPLEMENTATION_NAME` should be set to `paraview` and `CATALYST_IMPLEMENTATION_PATHS` should be set to the proper library in the ParaView folder.
```sh
export CATALYST_IMPLEMENTATION_PATHS="<path_to_downlaod_folder_of_paraview>/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/lib/catalyst"
export CATALYST_IMPLEMENTATION_NAME="ParaView"
```

I assume that you are familiar with the examples and manage to start them with catalyst.

# Tips and Tricks
## ParaView Live
In order to use the ParaView live feature with an existing Catalyst script the following options need to be set
```python
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'
```
After these settings you can start `ParaView -> Catalyst -> Connect`, accept the port and then ParaView is waiting for catalyst to connect.
Now also set the simulation to pause over `ParaView -> Catalyst -> Pause Simulation`, then start the simulation.
Now when it worked out you can see on Pipeline Browser of ParaView a small symbol left from catalyst stuff, this needs to be activated to then access the catalyst data.

## Creating a new script
Using ParaView Live you can quite easy get the data into the position and angle that you like for the simulation and depending on that you can then create different filter to exporting data.
Some choices do include dumping a csv, creating a vtp or also dumping images.

General workflow would be:
1. Add `CatalystAdaptor::Initialize()` directly after the MPI/IPPL initialize
2. Add `CatalystAdaptor::Execute.....()` where you want your data to be extracted
3. Add `CatalystAdaptor::Finalize()` directly before the MPI/IPPL finalize
4. Start simulation with matching channel names in live mode
5. Generate extraction / dump script
6. Pass the script as argument to the program executable
