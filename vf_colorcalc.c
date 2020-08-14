/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * scale video filter
 */
#include <stdio.h>
#include <string.h>

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "scale_eval.h"
#include "video.h"
#include "libavutil/avstring.h"
#include "libavutil/eval.h"
#include "libavutil/internal.h"
#include "libavutil/mathematics.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"
#include "libavutil/pixdesc.h"
#include "libavutil/imgutils.h"
#include "libavutil/avassert.h"
#include "libswscale/swscale.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, device_hwctx->internal->cuda_dl, x)

static const enum AVPixelFormat supported_formats[] = {
        AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_NV12,
        AV_PIX_FMT_YUV444P,
};

static const enum AVPixelFormat deinterleaved_formats[][2] = {
        { AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P },
};

enum ScaleStage {
    STAGE_DEINTERLEAVE,
    STAGE_RESIZE,
    STAGE_INTERLEAVE,
    STAGE_NB,
};

typedef struct NPPScaleStageContext {
    int stage_needed;
    enum AVPixelFormat in_fmt;
    enum AVPixelFormat out_fmt;

    struct {
        int width;
        int height;
    } planes_in[3], planes_out[3];

    AVBufferRef *frames_ctx;
    AVFrame     *frame;
} NPPScaleStageContext;

typedef struct ColorcaleContext {
    const AVClass *class;

    NPPScaleStageContext stages[STAGE_NB];
    AVFrame *tmp_frame;
    int passthrough;

    int shift_width, shift_height;

    /**
     * New dimensions. Special values are:
     *
     */
    int pix_fmt;

    /**
     * Output sw format. AV_PIX_FMT_NONE for no conversion.
     */
    enum AVPixelFormat format;

    char *w_expr;               ///< width  expression string
    char *h_expr;               ///< height expression string
    char *format_str;

    int force_original_aspect_ratio;
    int force_divisible_by;

    int interp_algo;
} ColorcaleContext;

static int init(AVFilterContext *ctx)
{
    ColorcaleContext *s = ctx->priv;
    int i;

    if (!strcmp(s->format_str, "same")) {
        s->format = AV_PIX_FMT_NONE;
    } else {
        s->format = av_get_pix_fmt(s->format_str);
        if (s->format == AV_PIX_FMT_NONE) {
            av_log(ctx, AV_LOG_ERROR, "Unrecognized pixel format: %s\n", s->format_str);
            return AVERROR(EINVAL);
        }
    }

    for (i = 0; i < FF_ARRAY_ELEMS(s->stages); i++) {
        s->stages[i].frame = av_frame_alloc();
        if (!s->stages[i].frame)
            return AVERROR(ENOMEM);
    }
    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static void uninit(AVFilterContext *ctx)
{
    ColorcaleContext *s = ctx->priv;
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(s->stages); i++) {
        av_frame_free(&s->stages[i].frame);
        av_buffer_unref(&s->stages[i].frames_ctx);
    }
    av_frame_free(&s->tmp_frame);
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
            AV_PIX_FMT_YUV420P,
            AV_PIX_FMT_YUV444P,
            AV_PIX_FMT_YUV440P,
            AV_PIX_FMT_YUV422P,
            AV_PIX_FMT_YUV411P,
            AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(ctx, pix_fmts);
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static enum AVPixelFormat get_deinterleaved_format(enum AVPixelFormat fmt)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(fmt);
    int i, planes;

    planes = av_pix_fmt_count_planes(fmt);
    if (planes == desc->nb_components)
        return fmt;
    for (i = 0; i < FF_ARRAY_ELEMS(deinterleaved_formats); i++)
        if (deinterleaved_formats[i][0] == fmt)
            return deinterleaved_formats[i][1];
    return AV_PIX_FMT_NONE;
}

static int colorcale_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext              *ctx = link->dst;
    ColorcaleContext                *s = ctx->priv;
    AVFilterLink             *outlink = ctx->outputs[0];
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)outlink->hw_frames_ctx->data;

    AVFrame* out = in;
    //the new output frame, property is the same as input frame
    av_log(NULL, AV_LOG_DEBUG, "vf_filter  time_base=%f \n", av_q2d(link->time_base));
    //_do_conversion(ctx, &td);
    fprintf(stdout, "--- wow first filter ---");
    fprintf(stdout, "%s", s->pix_fmt);
    return ff_filter_frame(outlink, out);
}

#define OFFSET(x) offsetof(ColorcaleContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
        { "pix_fmt",      "Output video pix_format",  OFFSET(w_expr),     AV_OPT_TYPE_STRING, { .str = NULL   }, .flags = FLAGS },
        { NULL },
};

static const AVClass colorcale_class = {
        .class_name = "colorcalc",
        .item_name  = av_default_item_name,
        .option     = options,
        .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad colorcalc_inputs[] = {
        {
                .name        = "default",
                .type        = AVMEDIA_TYPE_VIDEO,
                .filter_frame = colorcale_filter_frame,
        },
        { NULL }
};

static const AVFilterPad colorcalc_outputs[] = {
        {
                .name         = "default",
                .type         = AVMEDIA_TYPE_VIDEO,
                //.config_props = colorcale_config_props,
        },
        { NULL }
};

AVFilter ff_vf_colorcalc = {
        .name      = "colorcalc",
        .description = NULL_IF_CONFIG_SMALL("AAAAVVVV"),

        .init          = init,
        .uninit        = uninit,
        .query_formats = query_formats,

        .priv_size = sizeof(ColorcaleContext),
        .priv_class = &colorcale_class,

        .inputs    = colorcalc_inputs,
        .outputs   = colorcalc_outputs,

        .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};

