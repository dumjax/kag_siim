sudo nvidia-docker run -it -p 8888:8888 --shm-size 8G \
  -v `pwd`/data:/workdir/data \
  -v `pwd`/src:/workdir/src \
  -v `pwd`/notebooks:/workdir/notebooks \
  siim