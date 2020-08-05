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
[Ffmpeg](https://github.com/FFmpeg/FFmpeg)
[最简单的基于FFmpeg的封装格式处理](https://blog.csdn.net/leixiaohua1020/article/details/39802913)
[MPEG-2 TS - Format of a Transport Stream Packet](https://www.mikrocontroller.net/attachment/27265/mpeg2ts.pdf)
[mpegts-introduction](https://tsduck.io/download/docs/mpegts-introduction.pdf)
[Program-specific information](https://en.wikipedia.org/wiki/Program-specific_information)

### source code



## Learn it Hard way

### Chapter-1
using ffmepg API, remuxing .ts to .flv

### Chapter-2
ts file to flv in read/write bitfile

总的来说如何解析ts，就是找到 PAT，从 PAT 中找出对应存在的 program id，根据这些id找到 PMT，从中获到这些节目总的相对的媒体数据id，然后通过这些id，再从ts文件中找到这些文件的es数据，来完成解码或者别的什么操作。

#### ES:  

   ES--Elementary  Streams  (原始流)是直接从编码器出来的数据流，可以是编码过的视频数据流（H.264,MJPEG等），音频数据流（AAC），或其他编码数据流的统称。ES流经过PES打包器之后，被转换成PES包。

    ES是只包含一种内容的数据流，如只含视频或只含音频等，打包之后的PES也是只含一种性质的ES,如只含视频ES的PES,只含音频ES的PES等。每个ES都由若干个存取单元（AU）组成，每个视频AU或音频AU都是由头部和编码数据两部分组成，1个AU相当于编码的1幅视频图像或1个音频帧，也可以说，每个AU实际上是编码数据流的显示单元，即相当于解码的1幅视频图像或1个音频帧的取样。

#### PES:

    PES--Packetized  Elementary Streams  (分组的ES)，ES形成的分组称为PES分组，是用来传递ES的一种数据结构。PES流是ES流经过PES打包器处理后形成的数据流，在这个过程中完成了将ES流分组、打包、加入包头信息等操作（对ES流的第一次打包）。PES流的基本单位是PES包。PES包由包头和payload组成。

#### PTS、DTS:

   PTS--PresentationTime Stamp（显示时间标记）表示显示单元出现在系统目标解码器（H.264、MJPEG等）的时间。

   DTS--Decoding Time Stamp（解码时间标记）表示将存取单元全部字节从解码缓存器移走的时间。

   PTS/DTS是打在PES包的包头里面的，这两个参数是解决音视频同步显示，防止解码器输入缓存上溢或下溢的关键。每一个I（关键帧）、P（预测帧）、B（双向预测 帧）帧的包头都有一个PTS和DTS，但PTS与DTS对于B帧不一样，无需标出B帧的DTS，对于I帧和P帧，显示前一定要存储于视频解码器的重新排序缓存器中，经过延迟（重新排序）后再显示，所以一定要分别标明PTS和DTS。

#### PS:

   PS--Program Stream(节目流)PS流由PS包组成，而一个PS包又由若干个PES包组成（到这里，ES经过了两层的封装）。PS包的包头中包含了同步信息与时钟恢复信息。一个PS包最多可包含具有同一时钟基准的16个视频PES包和32个音频PES包。

#### TS:

    TS--Transport Stream（传输流）由定长的TS包组成（188字节），而TS包是对PES包的一个重新封装（到这里，ES也经过了两层的封装）。PES包的包头信息依然存在于TS包中。

    TS流与PS流的区别在于TS流的包结构是固定长度的,而PS流的包结构是可变长度的。PS包由于长度是变化的,一旦丢失某一PS包的同步信息,接收机就会进入失步状态,从而导致严重的信息丢失事件。而TS码流由于采用了固定长度的包结构,当传输误码破坏了某一TS包的同步信息时,接收机可在固定的位置检测它后面包中的同步信息,从而恢复同步,避免了信息丢失。因此在信道环境较为恶劣、传输误码较高时一般采用TS码流,而在信环境较好、传输误码较低时一般采用PS码流。

#### TS单一码流、混合码流:

  单一性：TS流的基本组成单位是长度为188字节的TS包。

  混合性： TS流由多种数据组合而成，一个TS包中的数据可以是视频数据，音频数据，填充数据，PSI/SI表格数据等（唯一的PID对应）。
