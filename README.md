# ffmpeg-learn-tutorial

[TOC]

## HOWTO
**env:**
```shell script
docker run -w /root --rm -it -v `pwd`:/root/work vonsago/ffmpeg-devel bash
```
**compile:**
```shell script
gcc -L/opt/ffmpeg/lib -I/opt/ffmpeg/include /work/0_hello_world.c \
 	  -lavcodec -lavformat -lavfilter -lavdevice -lswresample -lswscale -lavutil \
 	  -o /root/run_main
```
**run:**
```shell script
./root/run_main small_bunny_1080p_60fps.mp4 video.ts
./root/run_main video.ts video.flv
```
*docker image contains:* "ffmpeg source code", "flv_parser", "hexdump" and one ".mp4" file for test.
## Reference
### tutorial
[ffmpeg-libav-tutorial](https://github.com/vonsago/ffmpeg-libav-tutorial)
### blob
### source code
[Ffmpeg](https://github.com/FFmpeg/FFmpeg)
[最简单的基于FFmpeg的封装格式处理](https://blog.csdn.net/leixiaohua1020/article/details/39802913)


## Learn it Hard way

### Chapter-1


