## st_handeye_graph

To reproduce figures.

```bash
git clone https://github.com/koide3/st_handeye_graph.git
git clone https://bitbucket.org/koide3/st_handeye_eval.git

mkdir st_handeye_graph/build
cd st_handeye_graph/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../scripts
# run simulations and plot figures
./run_simulation.sh

# run calibration on real data and perform 3d reconstruction
./run_calibration.sh

# see the reconstructed point cloud
pcl_viewer data/points_3d_graph.pcd
```

### Paper
Kenji Koide and Emanuele Menegatti, General Hand-Eye Calibration based on Reprojection Error Minimization, IEEE Robotics and Automation Letters, (under view).
