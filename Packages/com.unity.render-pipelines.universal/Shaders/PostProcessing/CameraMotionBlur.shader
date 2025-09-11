Shader "Hidden/Universal Render Pipeline/CameraMotionBlur"
{
    HLSLINCLUDE
        #pragma vertex VertCMB
        #pragma fragment FragCMB
        #pragma multi_compile_fragment _ _ENABLE_ALPHA_OUTPUT

        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Random.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

#if defined(USING_STEREO_MATRICES)
            float4x4 _ViewProjMStereo[2];
            float4x4 _PrevViewProjMStereo[2];
#define _ViewProjM _ViewProjMStereo[unity_StereoEyeIndex]
#define _PrevViewProjM  _PrevViewProjMStereo[unity_StereoEyeIndex]
#else
        float4x4 _ViewProjM;
        float4x4 _PrevViewProjM;
#endif
        float _Intensity;
        float _Clamp;
        float4 _SourceSize;

        // TileMax filter parameters
        int _TileMaxLoop;
        float2 _TileMaxOffs;

        // Maximum blur radius (in pixels)
        float _MaxBlurRadius;
        float _RcpMaxBlurRadius;

        // Filter parameters/coefficients
        float _LoopCount;
        float _Separation;

        TEXTURE2D_X(_MainTex);
        TEXTURE2D_X(_MotionVectorTexture);
        TEXTURE2D_X(_VelocityTex);
        float4 _MotionVectorTexture_TexelSize;

        TEXTURE2D_X(_NeighborMaxTex);
        TEXTURE2D_X(_Tile2RT);
        TEXTURE2D_X(_Tile4RT);
        TEXTURE2D_X(_Tile8RT);
        TEXTURE2D_X(_TileVRT);
        float4 _Tile2RT_TexelSize;
        float4 _Tile4RT_TexelSize;
        float4 _Tile8RT_TexelSize;
        float4 _TileVRT_TexelSize;
        float4 _NeighborMaxTex_TexelSize;






        

        struct VaryingsCMB
        {
            float4 positionCS    : SV_POSITION;
            float4 texcoord      : TEXCOORD0;
            UNITY_VERTEX_OUTPUT_STEREO
        };

        VaryingsCMB VertCMB(Attributes input)
        {
            VaryingsCMB output;
            UNITY_SETUP_INSTANCE_ID(input);
            UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

            float4 pos = GetFullScreenTriangleVertexPosition(input.vertexID);
            float2 uv  = GetFullScreenTriangleTexCoord(input.vertexID);

            output.positionCS  = pos;
            output.texcoord.xy = DYNAMIC_SCALING_APPLY_SCALEBIAS(uv);

            float4 projPos = output.positionCS * 0.5;
            projPos.xy = projPos.xy + projPos.w;
            output.texcoord.zw = projPos.xy;

            return output;
        }

        float2 ClampVelocity(float2 velocity, float maxVelocity)
        {
            float len = length(velocity);
            return (len > 0.0) ? min(len, maxVelocity) * (velocity * rcp(len)) : 0.0;
        }

        float2 GetVelocity(float2 uv)
        {
            // Unity motion vectors are forward motion vectors in screen UV space
            float2 offsetUv = SAMPLE_TEXTURE2D_X(_MotionVectorTexture, sampler_LinearClamp, uv).xy;
            return -offsetUv;
        }

        // Per-pixel camera velocity
        float2 GetCameraVelocity(float4 uv)
        {
            #if UNITY_REVERSED_Z
                float depth = SampleSceneDepth(uv.xy).x;
            #else
                float depth = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(uv.xy).x);
            #endif

            float4 worldPos = float4(ComputeWorldSpacePosition(uv.xy, depth, UNITY_MATRIX_I_VP), 1.0);

            float4 prevClipPos = mul(_PrevViewProjM, worldPos);
            float4 curClipPos = mul(_ViewProjM, worldPos);

            float2 prevPosCS = prevClipPos.xy / prevClipPos.w;
            float2 curPosCS = curClipPos.xy / curClipPos.w;

            // Backwards motion vectors
            float2 velocity = (prevPosCS - curPosCS);
            #if UNITY_UV_STARTS_AT_TOP
                velocity.y = -velocity.y;
            #endif
            return ClampVelocity(velocity, _Clamp);
        }

        float4 GatherSample(float sampleNumber, float2 velocity, float invSampleCount, float2 centerUV, float randomVal, float velocitySign)
        {
            float  offsetLength = (sampleNumber + 0.5h) + (velocitySign * (randomVal - 0.5h));
            float2 sampleUV = centerUV + (offsetLength * invSampleCount) * velocity * velocitySign;


            #if UNITY_REVERSED_Z
                float Depth = SampleSceneDepth(centerUV.xy).x;
                float VelocityDepth = SampleSceneDepth(sampleUV.xy).x;

            #else
                float Depth = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(centerUV.xy).x);
                float VelocityDepth = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(sampleUV.xy).x);

            #endif

            float diff = VelocityDepth < Depth ? 1:0;

            return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_PointClamp, sampleUV); //* diff;
        }

        float4 DoMotionBlur(VaryingsCMB input, int iterations, int useMotionVectors)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            float2 velocity;
            if(useMotionVectors == 1)
            {
                velocity = ClampVelocity(GetVelocity(uv),_Clamp) * _Intensity;
                // Scale back to -1, 1 from 0..1 to match GetCameraVelocity. A workaround to keep existing visual look.
                // TODO: There's bug in GetCameraVelocity, which is using NDC and not UV
                velocity *= 2;
            }
            else
                velocity = GetCameraVelocity(float4(uv, input.texcoord.zw)) * _Intensity;

            float randomVal = InterleavedGradientNoise(uv * _SourceSize.xy, 0);
            float invSampleCount = rcp(iterations * 2.0);

            //float4 color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_PointClamp, uv);
            float4 color = 0;


            UNITY_UNROLL
            for (int i = 0; i < iterations; i++)
            {
                color += GatherSample(i, velocity, invSampleCount, uv, randomVal, -1.0);
                color += GatherSample(i, velocity, invSampleCount, uv, randomVal,  1.0);
            }

            #if _ENABLE_ALPHA_OUTPUT
                return color * invSampleCount;
            #else
                  // NOTE: Rely on the compiler to eliminate .w computation above
                return float4(color.xyz * invSampleCount, 1.0);
            #endif
        }

        float4 FragVelocitySetup(VaryingsCMB input)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            #if UNITY_REVERSED_Z
                float d = SampleSceneDepth(uv.xy).x;
            #else
                float d = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(uv.xy).x);
            #endif

            float2 v = SAMPLE_TEXTURE2D_X(_MotionVectorTexture, sampler_PointClamp, uv).xy;

            // Apply the exposure time and convert to the pixel space.
            v *= (_Intensity * 0.5) * _MotionVectorTexture_TexelSize.zw;

            // Clamp the vector with the maximum blur radius.
            v /= max(1.0, length(v) * _RcpMaxBlurRadius); //+ 0.01f;

            //v = clamp(v,-_Clamp * _MotionVectorTexture_TexelSize.zw,_Clamp * _MotionVectorTexture_TexelSize.zw);

            // Sample the depth of the pixel.
            d = Linear01Depth(d,_ZBufferParams);

            // Pack into 10/10/10/2 format.
            return float4((v * _RcpMaxBlurRadius + 1.0) * 0.5, d, 0.0);

        }

        float2 MaxV(float2 v1, float2 v2)
        {
            return dot(v1, v1) < dot(v2, v2) ? v2 : v1;
        }

        // TileMax filter (2 pixel width with normalization)
        float4 FragTileMax1(VaryingsCMB input)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            float4 d = (_MotionVectorTexture_TexelSize.xyxy * 0.5f) * float4(-0.5, -0.5, 0.5, 0.5);

            float2 v1 = SAMPLE_TEXTURE2D(_VelocityTex, sampler_PointClamp, uv + d.xy).rg;
            float2 v2 = SAMPLE_TEXTURE2D(_VelocityTex, sampler_PointClamp, uv + d.zy).rg;
            float2 v3 = SAMPLE_TEXTURE2D(_VelocityTex, sampler_PointClamp, uv + d.xw).rg;
            float2 v4 = SAMPLE_TEXTURE2D(_VelocityTex, sampler_PointClamp, uv + d.zw).rg;

            v1 = (v1 * 2.0 - 1.0) * _MaxBlurRadius;
            v2 = (v2 * 2.0 - 1.0) * _MaxBlurRadius;
            v3 = (v3 * 2.0 - 1.0) * _MaxBlurRadius;
            v4 = (v4 * 2.0 - 1.0) * _MaxBlurRadius;

            return float4(MaxV(MaxV(MaxV(v1, v2), v3), v4), 0.0, 0.0);
        }

        float4 FragTileMax2(VaryingsCMB input)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            float4 d = _Tile2RT_TexelSize.xyxy * float4(-0.5, -0.5, 0.5, 0.5);

            float2 v1 = SAMPLE_TEXTURE2D(_Tile2RT, sampler_PointClamp, uv + d.xy).rg;
            float2 v2 = SAMPLE_TEXTURE2D(_Tile2RT, sampler_PointClamp, uv + d.zy).rg;
            float2 v3 = SAMPLE_TEXTURE2D(_Tile2RT, sampler_PointClamp, uv + d.xw).rg;
            float2 v4 = SAMPLE_TEXTURE2D(_Tile2RT, sampler_PointClamp, uv + d.zw).rg;

            return float4(MaxV(MaxV(MaxV(v1, v2), v3), v4), 0.0, 0.0);
        }

        float4 FragTileMax4(VaryingsCMB input)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            float4 d = _Tile4RT_TexelSize.xyxy * float4(-0.5, -0.5, 0.5, 0.5);

            float2 v1 = SAMPLE_TEXTURE2D(_Tile4RT, sampler_PointClamp, uv + d.xy).rg;
            float2 v2 = SAMPLE_TEXTURE2D(_Tile4RT, sampler_PointClamp, uv + d.zy).rg;
            float2 v3 = SAMPLE_TEXTURE2D(_Tile4RT, sampler_PointClamp, uv + d.xw).rg;
            float2 v4 = SAMPLE_TEXTURE2D(_Tile4RT, sampler_PointClamp, uv + d.zw).rg;

            return float4(MaxV(MaxV(MaxV(v1, v2), v3), v4), 0.0, 0.0);
        }

        // TileMax filter (variable width)
        float4 FragTileMaxV(VaryingsCMB input)
        {
            float2 uv0 = input.texcoord + _Tile8RT_TexelSize.xy * _TileMaxOffs.xy;
            float2 du = float2(_Tile8RT_TexelSize.x, 0.0);
            float2 dv = float2(0.0, _Tile8RT_TexelSize.y);

            float2 vo = 0.0;

            UNITY_LOOP
            for (int ix = 0; ix < _TileMaxLoop; ix++)
            {
                UNITY_LOOP
                for (int iy = 0; iy < _TileMaxLoop; iy++)
                {
                    float2 uv = uv0 + du * ix + dv * iy;
                    vo = MaxV(vo, SAMPLE_TEXTURE2D(_Tile8RT, sampler_PointClamp, uv).rg);
                }
            }

            return float4(vo, 0.0, 0.0);
        }

        // NeighborMax filter
        float4 FragNeighborMax(VaryingsCMB input)
        {
            const float cw = 1.01; // Center weight tweak

            float4 d = _TileVRT_TexelSize.xyxy * float4(1.0, 1.0, -1.0, 0.0);

            float2 v1 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord - d.xy).rg;
            float2 v2 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord - d.wy).rg;
            float2 v3 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord - d.zy).rg;

            float2 v4 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord - d.xw).rg;
            float2 v5 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord).rg * cw;
            float2 v6 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord + d.xw).rg;

            float2 v7 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord + d.zy).rg;
            float2 v8 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord + d.wy).rg;
            float2 v9 = SAMPLE_TEXTURE2D(_TileVRT, sampler_PointClamp, input.texcoord + d.xy).rg;

            float2 va = MaxV(v1, MaxV(v2, v3));
            float2 vb = MaxV(v4, MaxV(v5, v6));
            float2 vc = MaxV(v7, MaxV(v8, v9));

            return float4(MaxV(va, MaxV(vb, vc)) * (1.0 / cw), 0.0, 0.0);
        }

        // Returns true or false with a given interval.
        bool Interval(float phase, float interval)
        {
            return frac(phase / interval) > 0.499;
        }

        float GradientNoise(float2 uv){

            uv = floor(uv * _ScreenParams.xy);
            float f = dot(float2(0.06711056, 0.00583715), uv);
            return frac(52.9829189 * frac(f));

            }

        float2 JitterTile(float2 uv)
        {
            float rx, ry;
            sincos(GradientNoise(uv + float2(2.0, 0.0)) * TWO_PI, ry, rx);
            return float2(rx, ry) * _NeighborMaxTex_TexelSize.xy * 0.25;
        }

        // Velocity sampling function
        float3 SampleVelocity(float2 uv)
        {
            // float3 v = SAMPLE_TEXTURE2D_LOD(_VelocityTex, sampler_LinearClamp, uv, 0.0).xyz;

            // v.xy = (v.xy * 2.0 - 1.0);
            // v.xy = v.xy * (1-step(length(v.xy),0.0f));
            // return float3(v.xy * _MaxBlurRadius, v.z);
            // //return float3((v.xy * 2.0 - 1.0) * _MaxBlurRadius, v.z);


            float3 v = SAMPLE_TEXTURE2D_LOD(_VelocityTex, sampler_PointClamp, uv, 0.0).xyz;
            return float3((v.xy * 2.0 - 1.0) * _MaxBlurRadius, v.z);

        }

        float4 FragReconstruction(VaryingsCMB input) : SV_Target
        {
            // Color sample at the center point
            const float4 c_p = SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearClamp, input.texcoord);

            // Velocity/Depth sample at the center point
            const float3 vd_p = SampleVelocity(input.texcoord);
            const float l_v_p = max(length(vd_p.xy), 0.5);
            const float rcp_d_p = 1.0 / vd_p.z;

            // NeighborMax vector sample at the center point
            const float2 v_max = SAMPLE_TEXTURE2D(_NeighborMaxTex, sampler_PointClamp, input.texcoord + JitterTile(input.texcoord)).xy;
            const float l_v_max = length(v_max);
            const float rcp_l_v_max = 1.0 / l_v_max;

            // Escape early if the NeighborMax vector is small enough.
            if (l_v_max < 2.0) return c_p;

            // Use V_p as a secondary sampling direction except when it's too small
            // compared to V_max. This vector is rescaled to be the length of V_max.
            const float2 v_alt = (l_v_p * 2.0 > l_v_max) ? vd_p.xy * (l_v_max / l_v_p) : v_max;

            // Determine the sample count.
            const float sc = floor(min(_LoopCount, l_v_max * 0.5));

            // Loop variables (starts from the outermost sample)
            const float dt = 1.0 / sc;
            const float t_offs = (GradientNoise(input.texcoord) - 0.5) * dt; //(InterleavedGradientNoise(input.texcoord * _SourceSize.xy,0) - 0.5) * dt;
            float t = 1.0 - dt * 0.5;
            float count = 0.0;

            // Background velocity
            // This is used for tracking the maximum velocity in the background layer.
            float l_v_bg = max(l_v_p, 1.0);

            // Color accumlation
            float4 acc = 0.0;

            UNITY_LOOP
            while (t > dt * 0.25)
            {
                // Sampling direction (switched per every two samples)
                const float2 v_s = Interval(count, 4.0) ? v_alt : v_max;

                // Sample position (inverted per every sample)
                const float t_s = (Interval(count, 2.0) ? -t : t) + t_offs;

                // Distance to the sample position
                const float l_t = l_v_max * abs(t_s);

                // UVs for the sample position
                const float2 uv0 = input.texcoord + v_s * t_s * _MotionVectorTexture_TexelSize.xy;
                const float2 uv1 = input.texcoord + v_s * t_s * _MotionVectorTexture_TexelSize.xy;

                // Color sample
                const float3 c = SAMPLE_TEXTURE2D(_BlitTexture, sampler_MirrorLinear, uv0).rgb;

                // Velocity/Depth sample
                const float3 vd = SampleVelocity(uv1);

                // Background/Foreground separation
                const float fg = saturate((vd_p.z - vd.z) * _Separation * rcp_d_p);

                // Length of the velocity vector
                const float l_v = lerp(l_v_bg, length(vd.xy), fg);

                // Sample weight
                // (Distance test) * (Spreading out by motion) * (Triangular window)
                const float w = saturate(l_v - l_t) / l_v * (1.2 - t);

                // Color accumulation
                acc += float4(c, 1.0) * w;

                // Update the background velocity.
                l_v_bg = max(l_v_bg, l_v);

                // Advance to the next sample.
                t = Interval(count, 2.0) ? t - dt : t;
                count += 1.0;
            }

            // Add the center sample.
            acc += float4(c_p.rgb, 1.0) * (1.2 / (l_v_bg * sc * 2.0));

            return float4(acc.rgb / (acc.a), c_p.a);
        }

    ENDHLSL

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline"}
        LOD 100
        ZTest Always ZWrite Off Cull Off

        // (0) Velocity texture setup
        Pass
        {
            Name "FragVelocitySetup"

            HLSLPROGRAM

                float4 FragCMB(VaryingsCMB input) : SV_Target
                {
                    return FragVelocitySetup(input);
                }

            ENDHLSL
        }

        // (1) TileMax filter (2 pixel width with normalization)
        Pass
        {
            Name "FragTileMax1"

            HLSLPROGRAM

                float4 FragCMB(VaryingsCMB input) : SV_Target
                {
                    return FragTileMax1(input);
                }

            ENDHLSL
        }

        //  (2) TileMax filter (2 pixel width)
        Pass
        {
            Name "FragTileMax2"

            HLSLPROGRAM

                float4 FragCMB(VaryingsCMB input) : SV_Target
                {
                    return FragTileMax2(input);
                }

            ENDHLSL
        }

        // (3) TileMax filter (variable width)
        Pass
        {
            Name "FragTileMaxV"

            HLSLPROGRAM

                float4 FragCMB(VaryingsCMB input) : SV_Target
                {
                    return FragTileMaxV(input);
                }

            ENDHLSL
        }

        // (4) NeighborMax filter
        Pass
        {
            Name "FragNeighborMax"

            HLSLPROGRAM

                float4 FragCMB(VaryingsCMB input) : SV_Target
                {
                    return FragNeighborMax(input);
                }

            ENDHLSL
        }

        // (5) Reconstruction filter
        Pass
        {
            Name "FragReconstruction"

            HLSLPROGRAM

                float4 FragCMB(VaryingsCMB input) : SV_Target
                {
                    return FragReconstruction(input);
                }

            ENDHLSL
        }

        //  (6) TileMax filter (2 pixel width) // extre
        Pass
        {
            Name "FragTileMax4"

            HLSLPROGRAM

                float4 FragCMB(VaryingsCMB input) : SV_Target
                {
                    return FragTileMax4(input);
                }

            ENDHLSL
        }
        
    }
}
