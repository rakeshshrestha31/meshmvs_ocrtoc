## Copy required data
```
cd <download_dir>
unzip meshmvs_ocrtoc_dataset.zip
mv meshmvs_ocrtoc_dataset <repo_dir>/datasets/ocrtoc/3d_dataset
unzip output_meshvoxdepth_gtdepth.zip
mv output_meshvoxdepth_gtdepth <repo_dir>/
mv ocrtoc_splits_val05.json <repo_dir>/datasets/ocrtoc/
```

## Start the container
```
xhost +local:'meshmvs'

docker run --privileged -h meshmvs --name meshmvs -it --cap-add=SYS_PTRACE \
   --net=host \
   --add-host meshmvs:127.0.0.1 \
   --env HOME=/home/${USER} \
   --env USER=${USER} \
   --env GROUP=${USER} \
   --env USER_ID=`id -u ${USER}` \
   --env GROUP_ID=`getent group ${USER} | awk -F: '{printf $$3}'` \
   --env EMAIL \
   --env GIT_AUTHOR_EMAIL \
   --env GIT_AUTHOR_NAME \
   --env GIT_COMMITTER_EMAIL \
   --env GIT_COMMITTER_NAME \
   --env SSH_AUTH_SOCK \
   --env TERM \
   --env DISPLAY \
   --env VIDEO_GROUP_ID=`getent group video | awk -F: '{printf $$3}'` \
   --volume <repo_dir>:/workspace \
   --volume /dev/dri:/dev/dri \
   --volume /dev/input:/dev/input \
   --volume /tmp/.X11-unix:/tmp/.X11-unix \
   --volume /dev/shm:/dev/shm \
   --volume /home/${USER}/.ssh:/home/${USER}/.ssh \
   --volume /run/user/`id -u ${USER}`/keyring/ssh:/run/user/`id -u ${USER}`/keyring/ssh \
   --gpus all \
   --env NVIDIA_VISIBLE_DEVICES=all \
   --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
   --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
   rakeshshrestha/meshmvs:latest bash
```

## Start training inside the container
```
cd /workspace
python tools/train_net_ocrtoc.py --config-file configs/ocrtoc/voxmesh_R50_depth.yaml OUTPUT_DIR output_meshvoxdepth_gtdepth
```
