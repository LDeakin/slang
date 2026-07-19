//TEST:SIMPLE(diag=CHECK_SPV_QUAD): -stage compute -entry computeMain -target spirv -DQUAD
//TEST:SIMPLE(diag=CHECK_SPV_QUAD): -stage compute -entry computeMain -target spirv -DQUAD -DTHREAD_COUNT_DEFAULT=3
//TEST:SIMPLE(diag=CHECK_SPV_LINEAR): -stage compute -entry computeMain -target spirv -DLINEAR
//TEST:SIMPLE(diag=CHECK_SPV_LINEAR): -stage compute -entry computeMain -target spirv -DLINEAR -DTHREAD_COUNT_DEFAULT=3
//TEST:SIMPLE(diag=CHECK_GLSL_QUAD): -stage compute -entry computeMain -target glsl -DQUAD
//TEST:SIMPLE(diag=CHECK_VIA_GLSL_QUAD): -stage compute -entry computeMain -target spirv -emit-spirv-via-glsl -DQUAD
//TEST:SIMPLE(diag=CHECK_VIA_GLSL_LINEAR): -stage compute -entry computeMain -target spirv -emit-spirv-via-glsl -DLINEAR
//TEST:SIMPLE(diag=CHECK_ERR): -stage compute -entry computeMain -target spirv -DQUAD_ODD_X

// A workgroup size given by specialization constants (SPIR-V `LocalSizeId`,
// GLSL `local_size_*_id`) is allowed with compute derivative groups; Vulkan
// validates the resolved size at pipeline creation time.
// The test runs each SPIR-V mode with a valid default of 8 and an invalid
// default of 3. Both must compile because the specialization value can change.

// CHECK_SPV_QUAD-DAG: OpExecutionModeId %computeMain LocalSizeId
// CHECK_SPV_QUAD-DAG: OpExecutionMode %computeMain DerivativeGroupQuads{{NV|KHR}}

// CHECK_SPV_LINEAR-DAG: OpExecutionModeId %computeMain LocalSizeId
// CHECK_SPV_LINEAR-DAG: OpExecutionMode %computeMain DerivativeGroupLinear{{NV|KHR}}

// CHECK_GLSL_QUAD-DAG: layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
// CHECK_GLSL_QUAD-DAG: layout(derivative_group_quadsNV) in;

// Verify that glslang also accepts spec-constant local size combined with a
// derivative-group layout qualifier.
// CHECK_VIA_GLSL_QUAD-DAG: OpExecutionMode %main DerivativeGroupQuads{{NV|KHR}}
// CHECK_VIA_GLSL_QUAD-DAG: SpecId 0

// CHECK_VIA_GLSL_LINEAR-DAG: OpExecutionMode %main DerivativeGroupLinear{{NV|KHR}}
// CHECK_VIA_GLSL_LINEAR-DAG: SpecId 0

#ifndef THREAD_COUNT_DEFAULT
#define THREAD_COUNT_DEFAULT 8
#endif

[vk::constant_id(0)]
const int threadCountX = THREAD_COUNT_DEFAULT;

[vk::constant_id(1)]
const int threadCountY = THREAD_COUNT_DEFAULT;

RWStructuredBuffer<float> outputBuffer;

#if defined(QUAD)
[DerivativeGroupQuad]
[numthreads(threadCountX, threadCountY, 1)]
#elif defined(LINEAR)
[DerivativeGroupLinear]
[numthreads(threadCountX, threadCountY, 1)]
#elif defined(QUAD_ODD_X)
// A literal axis that violates the quad constraint is still diagnosed even
// when another axis is a specialization constant.
[DerivativeGroupQuad]
[numthreads(3, threadCountY, 1)]
// CHECK_ERR: ([[#@LINE-1]]): error 31210: 
// CHECK_ERR:   [numthreads(3, threadCountY, 1)]
// CHECK_ERR:               ^
#endif
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
/*CHECK_ERR:
     ^^^^^^^^^^^ derivative group quad thread count error
     ^^^^^^^^^^^ compute derivative group quad requires thread dispatch count of X and Y to each be at a multiple of 2
*/
{
    outputBuffer[0] = ddx_fine(float(dispatchThreadID.x));
}
