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

## Colorspace
*ref:[ffmpeg-colorspace](https://trac.ffmpeg.org/wiki/colorspace), [YUV-wiki](https://wiki.videolan.org/YUV/) [yuv-wiki](https://en.wikipedia.org/wiki/YUV), [Chroma_subsampling-wiki](https://en.wikipedia.org/wiki/Chroma_subsampling)*
### Two concept
- RGB 将颜色像素分成了三个分量：红，绿，蓝
- YUV使用不同的表示模式来表示以下颜色的像素值：亮度（Y或亮度），色度（UV或色差）。 （注意：YUV代表3种组成）

*’YUV‘一词含糊不清，经常被错误地使用，包括FFmpeg中像素格式的定义。关于如何在数字视频中存储颜色的更准确的术语是YCbCr。另一方面，Y'UV指定由亮度（Y'）和色度（UV）分量组成的色彩空间。有关更多信息，请阅读相应的[文章](https://en.wikipedia.org/wiki/YUV)。在下文中，术语YUV与FFmpeg像素格式一样使用，指的是数字视频中的YCbCr。*

## ffmpeg filter guide

**格式规定**：

本文档在为`libavfilter`写`filters`时提供了指导。下文中"frame"指的是存储在AVFrame结构中的视频帧或一组音频样本。

`query_formats`方法应该规定每个输入和每个输出支持格式的列表。
对于视频，它表示`pixel format`，对于音频，它表示`channel layout`，sample format (the sample packing is implied by the sample format) and sample rate。

这些列表不仅是列表，它们是对共享库的引用。当协商机制计算链接两端支持的格式的交集时，对两个列表的所有引用都将替换为对交集的引用When the negotiation mechanism computes the intersection of the formats supported at each end of a link, all references to both lists are replaced  with a reference to the intersection。并且，当最终为其余列表之间的链接选择单一格式时，同样会更新对该列表的所有引用。

这意味着，如果过滤器要求其输入和输出在受支持的列表中具有相同的格式，则它所要做的就是使用对相同格式列表的引用。
`query_formats`可以保留某些格式，并返回AVERROR（EAGAIN）导致协商机制稍后重试。 具有复杂要求的过滤器可以使用该格式，以使用在一个链接上协商的格式来设置在另一个链接上支持的格式。

**原则**：

音频和视频数据量很大；'frame'和'frame'间的引用机制旨在尽可能避免该数据的多余副本，同时过滤器仍产生正确的结果。

数据存储在由AVFrame结构表示的缓冲区中。 几个引用可以指向同一个帧缓冲区。 一旦所有对应的引用都已销毁，缓冲区将自动释放。

数据的特征（分辨率，采样率等）存储在引用中。 同一缓冲区的不同引用可以显示不同的特性。 具体而言，一个视频引用只能指向视频缓冲区的一部分。通常将获得的引用作为filter_frame方法的输入，或者使用ff_get_video_buffer或ff_get_audio_buffer函数进行请求引用。 可以使用av_frame_ref()在现有缓冲区上创建新引用。 使用av_frame_free()函数销毁了引用。

**引用所有权**：

在任何时候，引用都“属于“特定的代码段，通常是过滤器。 在下面将要解释的一些警告中，仅允许该段代码访问它。 尽管有时这是自动完成的，但它也负责销毁它（请参阅有关链接参考字段的部分）。以下是（非常明显的）引用所有权的规则：
 * A reference received by the filter_frame method belongs to the   corresponding filter.
 * A reference passed to ff_filter_frame is given away and must no longer   be used.
 * A reference created with av_frame_ref() belongs to the code that   created it.
 * A reference obtained with ff_get_video_buffer or ff_get_audio_buffer   belongs to the code that requested it.
 * A reference given as return value by the get_video_buffer or   get_audio_buffer method is given away and must no longer be used.

**链路引用字段**：

AVFilterLink结构具有一些AVFrame字段。
partial_buf由libavfilter内部使用，并且不能由过滤器访问。
fifo包含在过滤器输入中排队的帧。直到被过滤器接受为止，它们都属于framework。

**引用权限**：

由于同一帧数据可以被多个帧共享，因此修改可能会带来意想不到的后果。 如果仅存在一个对frame的引用，则认为该frame可写。 拥有引用它的代码然后允许修改数据。

过滤器可以使用av_frame_is_writable()函数检查frame是否可写。
过滤器可以使用ff_inlink_make_frame_writable()函数来确保在代码的某些点可写frame。 如果需要，它将复制frame。
过滤器可以通过在相应的输入板上设置needs_writable标志来确保传递给filter_frame()回调的帧是可写的。 它不适用于activate()回调。

Frame scheduling:

这些规则的目的是确保帧在过滤器图形中流动而不会卡住或在某处累积。输入一帧输出一帧的简单过滤器不必担心。

过滤器有两种设计：一种使用filter_frame()和request_frame()回调，另一种使用activate()回调。第一种是旧设计，但是适用于具有单个输入并一次处理一帧的过滤器。 具有多个输入的新过滤器，一次处理多个帧，或者需要在EOF上进行特殊处理，应该使用带有activate()的设计。

**activate**:

当必须在过滤器中执行某些操作时，将调用此方法。 “内容”的定义取决于过滤器的语义。回调必须检查过滤器链接的状态，并进行相应的处理。
输出链接的状态存储在frame_wanted_out，status_in和status_out字段中，并由ff_outlink_frame_wanted()函数进行测试。如果此函数返回true，则处理过程需要此链接上的帧，并且希望过滤器朝该方向努力。
输入链接的状态通过status_in，fifo和status_out字段存储；不能直接访问它们。fifo字段包含在输入中排队等待过滤器处理的帧。status_in和status_out字段包含链接的排队状态（EOF或错误）。status_in是一种状态更改，在处理完fifo中的所有帧后必须将其考虑在内； status_out是已考虑的状态，如果不为0，则为最终状态。
 激活回调的典型任务是首先检查输出链接的后退状态，如果相关，则将其转发到相应的输入。然后，对于每个输入链接，如果相关，请：在fifo中测试帧的可用性并进行处理；如果没有可用的帧，则使用ff_inlink_acknowledge_status（）测试并确认状态变化；并将结果（框架或状态更改）转发到相应的输入。如果没有任何可能，请测试输出的状态并将其转发到相应的输入。如果仍然不可能，则返回FFERROR_NOT_READY。
如果过滤器在内部为某些输入存储一帧或几帧，则可以将其视为FIFO的一部分，并相应地延迟确认状态变化。
示例代码：
```c
ret = ff_outlink_get_status(outlink);
if (ret) {
    ff_inlink_set_status(inlink, ret);
    return 0;
}
if (priv->next_frame) {
    /* use it */
    return 0;
}
ret = ff_inlink_consume_frame(inlink, &frame);
if (ret < 0)
    return ret;
if (ret) {
    /* use it */
    return 0;
}
ret = ff_inlink_acknowledge_status(inlink, &status, &pts);
if (ret) {
    /* flush */
    ff_outlink_set_status(outlink, status, pts);
    return 0;
}
if (ff_outlink_frame_wanted(outlink)) {
    ff_inlink_request_frame(inlink);
    return 0;
}
return FFERROR_NOT_READY;
```

确切的代码取决于/ *use it* /块的相似程度以及它们与/ * flush * /块的关联程度，并且如果有多个，则需要将这些操作应用于正确的inlink或outlink。

可以使用宏来说明何时不需要额外的处理：

```c
FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);
FF_FILTER_FORWARD_STATUS_ALL(outlink, filter);
FF_FILTER_FORWARD_STATUS(inlink, outlink);
FF_FILTER_FORWARD_STATUS_ALL(inlink, filter);
FF_FILTER_FORWARD_WANTED(outlink, inlink);
```

**filter_frame**:

对于不使用activate()回调的过滤器，当将框架推入过滤器的输入时，将调用此方法。除非是以可重入的方式，可以在任何时候调用它。
如果输入帧足以产生输出，则过滤器应立即将输出帧推入输出链接。这里有个例外，如果输入帧足以产生多个输出帧，则过滤器每条链路仅需要输出至少一个。额外的帧可以留在过滤器中缓冲。如果新的输入产生新的输出，则必须立即清除这些缓冲的帧。（例如：帧速率加倍过滤器：filter_frame必须（1）刷新前一帧的第二个副本（如果仍然存在），（2）推送传入帧的第一个副本，（3）保留第二个副本以备后用）
如果输入帧不足以产生输出，则过滤器不得调用request_frame以获取更多信息。它必须仅处理帧或将其排队。请求更多帧的任务留给过滤器的request_frame方法或应用程序处理。
如果一个过滤器有多个输入，则该过滤器必须准备好在任何输入上随机到达的帧。因此，具有多个输入的任何滤波器很可能都需要某种排队机制。当输入太不平衡时，队列有限并且丢帧是完全可以接受的。

**request_frame**:

对于不使用activate()回调的过滤器，当在输出上需要一个帧时将调用此方法。
对于源，它应直接在相应的输出上调用filter_frame。
对于过滤器，如果已经准备好排队的帧，则应推送这些帧之一。如果没有，则过滤器应在其输入之一上请求一帧，直到至少推送了一帧为止。
返回值：如果request_frame可以生成框架，或者至少在生成框架方面取得进展，则应返回0；否则，返回0。如果由于暂时原因而无法执行，则应返回AVERROR（EAGAIN）;如果不能，因为没有更多的帧，则应返回AVERROR_EOF。
具有多个输入的过滤器的request_frame的典型实现如下所示：
```c
if (frames_queued) {
    push_one_frame();
    return 0;
}
input = input_where_a_frame_is_most_needed();
ret = ff_request_frame(input);
if (ret == AVERROR_EOF) {
    process_eof_on_input();
} else if (ret < 0) {
    return ret;
}
return 0;

```
 
请注意，除了可能具有排队的帧和源的过滤器之外，request_frame不会推送帧：它会将帧请求到其输入，并且作为响应，可能会调用filter_frame方法并完成工作。


### HOW-TO

[writing_filters](https://fossies.org/linux/ffmpeg/doc/writing_filters.txt)

- sed 's/edgedetect/foobar/g;s/EdgeDetect/Foobar/g' libavfilter/vf_edgedetect.c > libavfilter/vf_foobar.c
   20  - edit libavfilter/Makefile, and add an entry for "foobar" following the
   21    pattern of the other filters.
   22  - edit libavfilter/allfilters.c, and add an entry for "foobar" following the
   23    pattern of the other filters.
   24  - ./configure ...
   25  - make -j<whatever> ffmpeg
   26  - ./ffmpeg -i http://samples.ffmpeg.org/image-samples/lena.pnm -vf foobar foobar.png
   27    Note here: you can obviously use a random local image instead of a remote URL.
   
**Context:**

表示local state context，未初始化时，默认用0填充，所以不用担心读取出错。这里一般放一些需要的"global"信息，通常是存储用户选项的变量。如果有上下文，唯一需要注意的参数就是`const AVClass *class`。

**Options:**

定义了用户可以访问的选项，比如`-vf foobar=mode=colormix:high=0.4:low=0.1.`

have the following pattern:
   86   name, description, offset, type, default value, minimum value, maximum value, flags
   87 
   88  - name is the option name, keep it simple and lowercase
   89  - description are short, in lowercase, without period, and describe what they
   90    do, for example "set the foo of the bar"
   91  - offset is the offset of the field in your local context, see the OFFSET()
   92    macro; the option parser will use that information to fill the fields
   93    according to the user input
   94  - type is any of AV_OPT_TYPE_* defined in libavutil/opt.h
   95  - default value is an union where you pick the appropriate type; "{.dbl=0.3}",
   96    "{.i64=0x234}", "{.str=NULL}", ...
   97  - min and max values define the range of available values, inclusive
   98  - flags are AVOption generic flags. See AV_OPT_FLAG_* definitions
   99 
  100 When in doubt, just look at the other AVOption definitions all around the codebase,
  101 there are tons of examples.
  
## color space
* http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
### HDR

  	 * Nominal peak luminance (cd/m^2) for standard-dynamic range (SDR) systems.
	 *
	 * When a high dynamic range (HDR) transfer function is converted to linear
	 * light, the linear values are scaled such that nominal white (L = 1.0)
	 * matches the nominal SDR luminance. The HDR component of the signal is
	 * represented as multiples of the SDR luminance (L > 1.0).
	 *
	 * Certain HDR transfer functions (e.g. ST.2084) have a defined mapping
	 * between code values and physical luminance. When converting between
	 * absolute and relative transfer functions, the nominal peak luminance is
	 * used to scale the dequantized linear light values.

### bit depth
每一个RGB的色阶就是一条通道（channel），可以表达的范围就是bit depth，每条通道的bit depth被称为bpc

每一个像素的bit depth（像素各个通道bit depth之和）被称为 bpp

以三原色为例，一个像素有三个通道（RGB），那么bpp就是bpc的三倍

### ffmpeg build

https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
https://hhsprings.bitbucket.io/docs/programming/examples/ffmpeg/

### convert

https://www.vocal.com/video/rgb-and-yuv-color-space-conversion/

|   |BT.601|BT.709|BT.2020|
|:-:|-----:|-----:|------:|
| a | 0.299|0.2126| 0.2627|
| b | 0.587|0.7152| 0.6780|
| c | 0.114|0.0722| 0.0593|
| d | 1.772|1.8556| 1.8814|
| e | 1.402|1.5748| 1.4747|

* https://www.itu.int/rec/R-REC-BT.601
* https://www.itu.int/rec/R-REC-BT.709
* https://www.itu.int/rec/R-REC-BT.2020