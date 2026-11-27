/* Keep the network and tensor buffers open across frames. */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vip_lite.h>

typedef struct {
    vip_network network;
    vip_buffer input;
    vip_buffer output;
    size_t input_bytes;
    size_t output_bytes;
    bool initialized;
    bool prepared;
} Segmenter;

#define CHECK_VIP(call)                                                      \
    do {                                                                     \
        vip_status_e status = (call);                                        \
        if (status != VIP_SUCCESS) {                                         \
            snprintf(error, error_size, "%s: VIPLite status %d", #call, status); \
            goto failed;                                                     \
        }                                                                    \
    } while (0)

static bool create_tensor_buffer(
    vip_network network, bool output, vip_uint32_t width, vip_uint32_t height,
    vip_uint32_t channels, vip_buffer *buffer, float *scale, int32_t *zero_point,
    char *error, size_t error_size
) {
    vip_uint32_t count = 0;
    vip_buffer_create_params_t params = {0};
    const char *name = output ? "output" : "input";
    vip_network_property_e count_property =
        output ? VIP_NETWORK_PROP_OUTPUT_COUNT : VIP_NETWORK_PROP_INPUT_COUNT;
    vip_status_e (*query)(vip_network, vip_uint32_t, vip_enum, void *) =
        output ? vip_query_output : vip_query_input;

    CHECK_VIP(vip_query_network(network, count_property, &count));
    if (count != 1) {
        snprintf(error, error_size, "Expected one NPU %s tensor, got %u", name, count);
        return false;
    }
    CHECK_VIP(query(network, 0, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &params.num_of_dims));
    CHECK_VIP(query(network, 0, VIP_BUFFER_PROP_DATA_FORMAT, &params.data_format));
    CHECK_VIP(query(network, 0, VIP_BUFFER_PROP_QUANT_FORMAT, &params.quant_format));
    if (params.num_of_dims != 4 || params.data_format != VIP_BUFFER_FORMAT_INT8 ||
        params.quant_format != VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        snprintf(error, error_size, "Expected a four-dimensional affine INT8 %s", name);
        return false;
    }
    CHECK_VIP(query(network, 0, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, params.sizes));
    if (params.sizes[0] != width || params.sizes[1] != height ||
        params.sizes[2] != channels || params.sizes[3] != 1) {
        snprintf(error, error_size, "Expected NPU %s shape 1x%ux%ux%u",
                 name, channels, height, width);
        return false;
    }
    CHECK_VIP(query(network, 0, VIP_BUFFER_PROP_TF_SCALE, &params.quant_data.affine.scale));
    CHECK_VIP(query(network, 0, VIP_BUFFER_PROP_TF_ZERO_POINT,
                    &params.quant_data.affine.zeroPoint));
    float tensor_scale = params.quant_data.affine.scale;
    int32_t tensor_zero_point = params.quant_data.affine.zeroPoint;
    if (!isfinite(tensor_scale) || tensor_scale <= 0 ||
        tensor_zero_point < INT8_MIN || tensor_zero_point > INT8_MAX) {
        snprintf(error, error_size, "Invalid NPU %s quantization", name);
        return false;
    }
    CHECK_VIP(vip_create_buffer(&params, sizeof(params), buffer));
    if (vip_get_buffer_size(*buffer) < (size_t)width * height * channels) {
        snprintf(error, error_size, "NPU %s buffer is smaller than its tensor", name);
        return false;
    }
    *scale = tensor_scale;
    *zero_point = tensor_zero_point;
    return true;

failed:
    return false;
}

void smartcane_viplite_close(Segmenter *segmenter)
{
    if (segmenter->prepared) {
        vip_finish_network(segmenter->network);
    }
    if (segmenter->network) {
        vip_destroy_network(segmenter->network);
    }
    if (segmenter->input) {
        vip_destroy_buffer(segmenter->input);
    }
    if (segmenter->output) {
        vip_destroy_buffer(segmenter->output);
    }
    if (segmenter->initialized) {
        vip_destroy();
    }
    free(segmenter);
}

Segmenter *smartcane_viplite_open(
    const char *path, uint32_t width, uint32_t height, uint32_t classes,
    float *input_scale, int32_t *input_zero_point, char *error, size_t error_size
) {
    Segmenter *segmenter = calloc(1, sizeof(*segmenter));
    if (!segmenter) {
        snprintf(error, error_size, "Could not allocate VIPLite session");
        return NULL;
    }
    CHECK_VIP(vip_init());
    segmenter->initialized = true;
    CHECK_VIP(vip_create_network(path, 0, VIP_CREATE_NETWORK_FROM_FILE,
                                 &segmenter->network));

    float output_scale;
    int32_t output_zero_point;
    if (!create_tensor_buffer(segmenter->network, false, width, height, 3,
                              &segmenter->input, input_scale, input_zero_point,
                              error, error_size) ||
        !create_tensor_buffer(segmenter->network, true, width, height, classes,
                              &segmenter->output, &output_scale, &output_zero_point,
                              error, error_size)) {
        goto failed;
    }
    segmenter->input_bytes = (size_t)width * height * 3;
    segmenter->output_bytes = (size_t)width * height * classes;
    CHECK_VIP(vip_prepare_network(segmenter->network));
    segmenter->prepared = true;
    CHECK_VIP(vip_set_input(segmenter->network, 0, segmenter->input));
    CHECK_VIP(vip_set_output(segmenter->network, 0, segmenter->output));
    return segmenter;

failed:
    smartcane_viplite_close(segmenter);
    return NULL;
}

bool smartcane_viplite_run(
    Segmenter *segmenter, const int8_t *input, int8_t *output,
    char *error, size_t error_size
) {
    void *buffer = vip_map_buffer(segmenter->input);
    if (!buffer) {
        snprintf(error, error_size, "Could not map the NPU input buffer");
        return false;
    }
    memcpy(buffer, input, segmenter->input_bytes);
    CHECK_VIP(vip_unmap_buffer(segmenter->input));
    CHECK_VIP(vip_flush_buffer(segmenter->input, VIP_BUFFER_OPER_TYPE_FLUSH));
    CHECK_VIP(vip_run_network(segmenter->network));
    CHECK_VIP(vip_flush_buffer(segmenter->output, VIP_BUFFER_OPER_TYPE_INVALIDATE));
    buffer = vip_map_buffer(segmenter->output);
    if (!buffer) {
        snprintf(error, error_size, "Could not map the NPU output buffer");
        return false;
    }
    memcpy(output, buffer, segmenter->output_bytes);
    CHECK_VIP(vip_unmap_buffer(segmenter->output));
    return true;

failed:
    return false;
}
