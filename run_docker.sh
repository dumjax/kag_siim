sudo nvidia-docker run -it -p 8888:8888 \
  -v `pwd`/data:/workdir/data \
  -v `pwd`/src:/workdir/src \
  -v `pwd`/notebooks:/workdir/notebooks \
  siim
