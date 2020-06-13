sudo nvidia-docker run -it -p 8888:8888 --shm-size 8G \
  -v `pwd`/data:/workdir/data \
  -v `pwd`/src:/workdir/src \
  -v `pwd`/notebooks:/workdir/notebooks \
  -v `pwd`/models:/workdir/models \
  -v `pwd`/logs:/wordir/logs \
  siim
