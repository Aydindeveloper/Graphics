//-------------------------------------------------------------------------------------
// FragInputs
// This structure gather all possible varying/interpolator for this shader.
//-------------------------------------------------------------------------------------

#include "Packages/com.unity.render-pipelines.high-definition/Runtime/Debug/MaterialDebug.cs.hlsl"

#ifndef FRAG_INPUTS_ENABLE_STRIPPING
    #define FRAG_INPUTS_USE_TEXCOORD0
    #define FRAG_INPUTS_USE_TEXCOORD1
    #define FRAG_INPUTS_USE_TEXCOORD2
    #define FRAG_INPUTS_USE_TEXCOORD3
    #define FRAG_INPUTS_USE_TEXCOORD4
    #define FRAG_INPUTS_USE_TEXCOORD5
    #define FRAG_INPUTS_USE_TEXCOORD6
    #define FRAG_INPUTS_USE_TEXCOORD7
#endif

struct FragInputs
{
    // Contain value return by SV_POSITION (That is name positionCS in PackedVarying).
    // xy: unormalized screen position (offset by 0.5), z: device depth, w: depth in view space
    // Note: SV_POSITION is the result of the clip space position provide to the vertex shaders that is transform by the viewport
    float4 positionSS; // In case depth offset is use, positionRWS.w is equal to depth offset
    float3 positionRWS; // Relative camera space position
    float3 positionPredisplacementRWS; // Relative camera space position
    float2 positionPixel;              // Pixel position (VPOS)

    #ifdef FRAG_INPUTS_USE_TEXCOORD0
        float4 texCoord0;
    #endif

    #ifdef FRAG_INPUTS_USE_TEXCOORD1
        float4 texCoord1;
    #endif

    #ifdef FRAG_INPUTS_USE_TEXCOORD2
        float4 texCoord2;
    #endif

    #ifdef FRAG_INPUTS_USE_TEXCOORD3
        float4 texCoord3;
    #endif

    #ifdef FRAG_INPUTS_USE_TEXCOORD4
        float4 texCoord4;
    #endif

    #ifdef FRAG_INPUTS_USE_TEXCOORD5
        float4 texCoord5;
    #endif

    #ifdef FRAG_INPUTS_USE_TEXCOORD6
        float4 texCoord6;
    #endif

    #ifdef FRAG_INPUTS_USE_TEXCOORD7
        float4 texCoord7;
    #endif

    #ifdef FRAG_INPUTS_USE_INSTANCEID
        uint instanceID;
    #endif

    #ifdef FRAG_INPUTS_USE_SIX_WAY_DIFFUSE_GI_DATA
        float4 diffuseGIData[3];
    #endif

    float4 color; // vertex color

    // TODO: confirm with Morten following statement
    // Our TBN is orthogonal but is maybe not orthonormal in order to be compliant with external bakers (Like xnormal that use mikktspace).
    // (xnormal for example take into account the interpolation when baking the normal and normalizing the tangent basis could cause distortion).
    // When using tangentToWorld with surface gradient, it doesn't normalize the tangent/bitangent vector (We instead use exact same scale as applied to interpolated vertex normal to avoid breaking compliance).
    // this mean that any usage of tangentToWorld[1] or tangentToWorld[2] outside of the context of normal map (like for POM) must normalize the TBN (TCHECK if this make any difference ?)
    // When not using surface gradient, each vector of tangentToWorld are normalize (TODO: Maybe they should not even in case of no surface gradient ? Ask Morten)
    float3x3 tangentToWorld;

    uint primitiveID; // Only with fullscreen pass debug currently - not supported on all platforms

    // For two sided lighting
    bool isFrontFace;

    // append a substruct for custom interpolators to be copied correctly into SDI from Varyings.
    #if defined(USE_CUSTOMINTERP_SUBSTRUCT)
        CustomInterpolators customInterpolators;
    #endif

    // Append an additional substruct for VFX interpolators. Eventually, we should merge this with custom interpolators.
    #if defined(HAVE_VFX_MODIFICATION)
        FragInputsVFX vfx;
    #endif
};

void GetVaryingsDataDebug(uint paramId, FragInputs input, inout float3 result, inout bool needLinearToSRGB)
{
    switch (paramId)
    {
    #ifdef FRAG_INPUTS_USE_TEXCOORD0
        case DEBUGVIEWVARYING_TEXCOORD0:
            result = input.texCoord0.xyz;
            break;
    #endif
    #ifdef FRAG_INPUTS_USE_TEXCOORD1
        case DEBUGVIEWVARYING_TEXCOORD1:
            result = input.texCoord1.xyz;
            break;
    #endif
    #ifdef FRAG_INPUTS_USE_TEXCOORD2
        case DEBUGVIEWVARYING_TEXCOORD2:
            result = input.texCoord2.xyz;
            break;
    #endif
    #ifdef FRAG_INPUTS_USE_TEXCOORD3
        case DEBUGVIEWVARYING_TEXCOORD3:
            result = input.texCoord3.xyz;
            break;
    #endif
    #ifdef FRAG_INPUTS_USE_TEXCOORD4
        case DEBUGVIEWVARYING_TEXCOORD4:
            result = input.texCoord4.xyz;
            break;
    #endif
    #ifdef FRAG_INPUTS_USE_TEXCOORD5
        case DEBUGVIEWVARYING_TEXCOORD5:
            result = input.texCoord5.xyz;
            break;
    #endif
    #ifdef FRAG_INPUTS_USE_TEXCOORD6
        case DEBUGVIEWVARYING_TEXCOORD6:
            result = input.texCoord6.xyz;
            break;
    #endif
    #ifdef FRAG_INPUTS_USE_TEXCOORD7
        case DEBUGVIEWVARYING_TEXCOORD7:
            result = input.texCoord7.xyz;
            break;
    #endif
    case DEBUGVIEWVARYING_VERTEX_TANGENT_WS:
        result = input.tangentToWorld[0].xyz * 0.5 + 0.5;
        break;
    case DEBUGVIEWVARYING_VERTEX_BITANGENT_WS:
        result = input.tangentToWorld[1].xyz * 0.5 + 0.5;
        break;
    case DEBUGVIEWVARYING_VERTEX_NORMAL_WS:
        result = IsNormalized(input.tangentToWorld[2].xyz) ?  input.tangentToWorld[2].xyz * 0.5 + 0.5 : float3(1.0, 0.0, 0.0);
        break;
    case DEBUGVIEWVARYING_VERTEX_COLOR:
        result = input.color.rgb; needLinearToSRGB = true;
        break;
    case DEBUGVIEWVARYING_VERTEX_COLOR_ALPHA:
        result = input.color.aaa;
        break;
    }
}

void AdjustFragInputsToOffScreenRendering(inout FragInputs input, bool offScreenRenderingEnabled, float offScreenRenderingFactor)
{
    // We need to readapt the SS position as our screen space positions are for a low res buffer, but we try to access a full res buffer.
    input.positionSS.xy = offScreenRenderingEnabled ? (uint2)round(input.positionSS.xy * offScreenRenderingFactor) : input.positionSS.xy;
    input.positionPixel = offScreenRenderingEnabled ? (uint2)round(input.positionPixel * offScreenRenderingFactor) : input.positionPixel;
}
