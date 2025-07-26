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
        half _Intensity;
        half _Clamp;
        half4 _SourceSize;

        // TileMax filter parameters
        int _TileMaxLoop;
        half2 _TileMaxOffs;

        // Maximum blur radius (in pixels)
        half _MaxBlurRadius;
        half _RcpMaxBlurRadius;

        // Filter parameters/coefficients
        half _LoopCount;
        half _Separation;

        TEXTURE2D_X(_MainTex);
        TEXTURE2D_X(_MotionVectorTexture);
        TEXTURE2D_X(_VelocityTex);
        half4 _MotionVectorTexture_TexelSize;

        TEXTURE2D_X(_NeighborMaxTex);
        TEXTURE2D_X(_Tile2RT);
        TEXTURE2D_X(_Tile4RT);
        TEXTURE2D_X(_Tile8RT);
        TEXTURE2D_X(_TileVRT);
        half4 _Tile2RT_TexelSize;
        half4 _Tile4RT_TexelSize;
        half4 _Tile8RT_TexelSize;
        half4 _TileVRT_TexelSize;
        half4 _NeighborMaxTex_TexelSize;






        

        struct VaryingsCMB
        {
            half4 positionCS    : SV_POSITION;
            half4 texcoord      : TEXCOORD0;
            UNITY_VERTEX_OUTPUT_STEREO
        };

        VaryingsCMB VertCMB(Attributes input)
        {
            VaryingsCMB output;
            UNITY_SETUP_INSTANCE_ID(input);
            UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

            half4 pos = GetFullScreenTriangleVertexPosition(input.vertexID);
            half2 uv  = GetFullScreenTriangleTexCoord(input.vertexID);

            output.positionCS  = pos;
            output.texcoord.xy = DYNAMIC_SCALING_APPLY_SCALEBIAS(uv);

            half4 projPos = output.positionCS * 0.5;
            projPos.xy = projPos.xy + projPos.w;
            output.texcoord.zw = projPos.xy;

            return output;
        }

        half2 ClampVelocity(half2 velocity, half maxVelocity)
        {
            half len = length(velocity);
            return (len > 0.0) ? min(len, maxVelocity) * (velocity * rcp(len)) : 0.0;
        }

        half2 GetVelocity(half2 uv)
        {
            // Unity motion vectors are forward motion vectors in screen UV space
            half2 offsetUv = SAMPLE_TEXTURE2D_X(_MotionVectorTexture, sampler_LinearClamp, uv).xy;
            return -offsetUv;
        }

        // Per-pixel camera velocity
        half2 GetCameraVelocity(half4 uv)
        {
            #if UNITY_REVERSED_Z
                half depth = SampleSceneDepth(uv.xy).x;
            #else
                half depth = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(uv.xy).x);
            #endif

            half4 worldPos = half4(ComputeWorldSpacePosition(uv.xy, depth, UNITY_MATRIX_I_VP), 1.0);

            half4 prevClipPos = mul(_PrevViewProjM, worldPos);
            half4 curClipPos = mul(_ViewProjM, worldPos);

            half2 prevPosCS = prevClipPos.xy / prevClipPos.w;
            half2 curPosCS = curClipPos.xy / curClipPos.w;

            // Backwards motion vectors
            half2 velocity = (prevPosCS - curPosCS);
            #if UNITY_UV_STARTS_AT_TOP
                velocity.y = -velocity.y;
            #endif
            return ClampVelocity(velocity, _Clamp);
        }

        half4 GatherSample(half sampleNumber, half2 velocity, half invSampleCount, half2 centerUV, half randomVal, half velocitySign)
        {
            half  offsetLength = (sampleNumber + 0.5h) + (velocitySign * (randomVal - 0.5h));
            half2 sampleUV = centerUV + (offsetLength * invSampleCount) * velocity * velocitySign;


            #if UNITY_REVERSED_Z
                half Depth = SampleSceneDepth(centerUV.xy).x;
                half VelocityDepth = SampleSceneDepth(sampleUV.xy).x;

            #else
                half Depth = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(centerUV.xy).x);
                half VelocityDepth = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(sampleUV.xy).x);

            #endif

            half diff = VelocityDepth < Depth ? 1:0;

            return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_PointClamp, sampleUV); //* diff;
        }

        half4 DoMotionBlur(VaryingsCMB input, int iterations, int useMotionVectors)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            half2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            half2 velocity;
            if(useMotionVectors == 1)
            {
                velocity = ClampVelocity(GetVelocity(uv),_Clamp) * _Intensity;
                // Scale back to -1, 1 from 0..1 to match GetCameraVelocity. A workaround to keep existing visual look.
                // TODO: There's bug in GetCameraVelocity, which is using NDC and not UV
                velocity *= 2;
            }
            else
                velocity = GetCameraVelocity(half4(uv, input.texcoord.zw)) * _Intensity;

            half randomVal = InterleavedGradientNoise(uv * _SourceSize.xy, 0);
            half invSampleCount = rcp(iterations * 2.0);

            //half4 color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_PointClamp, uv);
            half4 color = 0;


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
                return half4(color.xyz * invSampleCount, 1.0);
            #endif
        }

        half4 FragVelocitySetup(VaryingsCMB input)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            half2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            #if UNITY_REVERSED_Z
                half d = SampleSceneDepth(uv.xy).x;
            #else
                half d = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(uv.xy).x);
            #endif

            half2 v = SAMPLE_TEXTURE2D_X(_MotionVectorTexture, sampler_LinearClamp, uv).xy;

            // Apply the exposure time and convert to the pixel space.
            v *= (_Intensity * 0.5) * _MotionVectorTexture_TexelSize.zw;

            // Clamp the vector with the maximum blur radius.
            v /= max(1.0, length(v) * 0.0185185); //+ 0.01f;

            //v = clamp(v,-_Clamp * _MotionVectorTexture_TexelSize.zw,_Clamp * _MotionVectorTexture_TexelSize.zw);

            // Sample the depth of the pixel.
            d = Linear01Depth(d,_ZBufferParams);

            // Pack into 10/10/10/2 format.
            return half4((v * 0.0185185 + 1.0h) * 0.5h, d, 0.0);

        }

        half2 MaxV(half2 v1, half2 v2)
        {
            return dot(v1, v1) < dot(v2, v2) ? v2 : v1;
        }

        // TileMax filter (2 pixel width with normalization)
        half4 FragTileMax1(VaryingsCMB input)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            half2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            half4 d = _MotionVectorTexture_TexelSize.xyxy * half4(-0.5, -0.5, 0.5, 0.5);

            half2 v1 = SAMPLE_TEXTURE2D(_VelocityTex, sampler_LinearClamp, uv + d.xy).rg;
            half2 v2 = SAMPLE_TEXTURE2D(_VelocityTex, sampler_LinearClamp, uv + d.zy).rg;
            half2 v3 = SAMPLE_TEXTURE2D(_VelocityTex, sampler_LinearClamp, uv + d.xw).rg;
            half2 v4 = SAMPLE_TEXTURE2D(_VelocityTex, sampler_LinearClamp, uv + d.zw).rg;

            v1 = (v1 * 2.0 - 1.0) * _MaxBlurRadius;
            v2 = (v2 * 2.0 - 1.0) * _MaxBlurRadius;
            v3 = (v3 * 2.0 - 1.0) * _MaxBlurRadius;
            v4 = (v4 * 2.0 - 1.0) * _MaxBlurRadius;

            return half4(MaxV(MaxV(MaxV(v1, v2), v3), v4), 0.0, 0.0);
        }

        half4 FragTileMax2(VaryingsCMB input)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            half2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            half4 d = _Tile2RT_TexelSize.xyxy * half4(-0.5, -0.5, 0.5, 0.5);

            half2 v1 = SAMPLE_TEXTURE2D(_Tile2RT, sampler_LinearClamp, uv + d.xy).rg;
            half2 v2 = SAMPLE_TEXTURE2D(_Tile2RT, sampler_LinearClamp, uv + d.zy).rg;
            half2 v3 = SAMPLE_TEXTURE2D(_Tile2RT, sampler_LinearClamp, uv + d.xw).rg;
            half2 v4 = SAMPLE_TEXTURE2D(_Tile2RT, sampler_LinearClamp, uv + d.zw).rg;

            return half4(MaxV(MaxV(MaxV(v1, v2), v3), v4), 0.0, 0.0);
        }

        half4 FragTileMax4(VaryingsCMB input)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            half2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord.xy);

            half4 d = _Tile4RT_TexelSize.xyxy * half4(-0.5, -0.5, 0.5, 0.5);

            half2 v1 = SAMPLE_TEXTURE2D(_Tile4RT, sampler_LinearClamp, uv + d.xy).rg;
            half2 v2 = SAMPLE_TEXTURE2D(_Tile4RT, sampler_LinearClamp, uv + d.zy).rg;
            half2 v3 = SAMPLE_TEXTURE2D(_Tile4RT, sampler_LinearClamp, uv + d.xw).rg;
            half2 v4 = SAMPLE_TEXTURE2D(_Tile4RT, sampler_LinearClamp, uv + d.zw).rg;

            return half4(MaxV(MaxV(MaxV(v1, v2), v3), v4), 0.0, 0.0);
        }

        // TileMax filter (variable width)
        half4 FragTileMaxV(VaryingsCMB input)
        {
            half2 uv0 = input.texcoord + _Tile8RT_TexelSize.xy * _TileMaxOffs.xy;
            half2 du = half2(_Tile8RT_TexelSize.x, 0.0);
            half2 dv = half2(0.0, _Tile8RT_TexelSize.y);

            half2 vo = 0.0;

            UNITY_LOOP
            for (int ix = 0; ix < _TileMaxLoop; ix++)
            {
                UNITY_LOOP
                for (int iy = 0; iy < _TileMaxLoop; iy++)
                {
                    half2 uv = uv0 + du * ix + dv * iy;
                    vo = MaxV(vo, SAMPLE_TEXTURE2D(_Tile8RT, sampler_LinearClamp, uv).rg);
                }
            }

            return half4(vo, 0.0, 0.0);
        }

        // NeighborMax filter
        half4 FragNeighborMax(VaryingsCMB input)
        {
            const half cw = 1.01; // Center weight tweak

            half4 d = _TileVRT_TexelSize.xyxy * half4(1.0, 1.0, -1.0, 0.0);

            half2 v1 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord - d.xy).rg;
            half2 v2 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord - d.wy).rg;
            half2 v3 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord - d.zy).rg;

            half2 v4 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord - d.xw).rg;
            half2 v5 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord).rg * cw;
            half2 v6 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord + d.xw).rg;

            half2 v7 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord + d.zy).rg;
            half2 v8 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord + d.wy).rg;
            half2 v9 = SAMPLE_TEXTURE2D(_TileVRT, sampler_LinearClamp, input.texcoord + d.xy).rg;

            half2 va = MaxV(v1, MaxV(v2, v3));
            half2 vb = MaxV(v4, MaxV(v5, v6));
            half2 vc = MaxV(v7, MaxV(v8, v9));

            return half4(MaxV(va, MaxV(vb, vc)) * (1.0 / cw), 0.0, 0.0);
        }

        // Returns true or false with a given interval.
        bool Interval(half phase, half interval)
        {
            return frac(phase / interval) > 0.499;
        }

        float2 JitterTile(float2 uv)
        {
            //half rx, ry;
            //sincos(GradientNoise(uv + half2(2.0, 0.0)) * TWO_PI, ry, rx);
            //return half2(rx, ry) * _NeighborMaxTex_TexelSize.xy * 0.25;

            return half2(InterleavedGradientNoise(uv * _NeighborMaxTex_TexelSize.xy, 0),InterleavedGradientNoise(uv * _NeighborMaxTex_TexelSize.xy, 1)) * _NeighborMaxTex_TexelSize.xy * 0.25;
        }

        // Velocity sampling function
        half3 SampleVelocity(half2 uv)
        {
            half3 v = SAMPLE_TEXTURE2D_LOD(_VelocityTex, sampler_LinearClamp, uv, 0.0).xyz;

            //v.xy = (v.xy * 2.0 - 1.0);
            //v.xy = v.xy * (1-step(length(v.xy),0f));
            //return half3(v.xy * _MaxBlurRadius, v.z);
            return half3((v.xy * 2.0 - 1.0) * _MaxBlurRadius, v.z);

        }

        half4 FragReconstruction(VaryingsCMB input) : SV_Target
        {
            // Color sample at the center point
            const half4 c_p = SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearClamp, input.texcoord);

            // Velocity/Depth sample at the center point
            const half3 vd_p = SampleVelocity(input.texcoord);
            const half l_v_p = max(length(vd_p.xy), 0.5);
            //const half rcp_d_p = 1.0 / vd_p.z;
            const half rcp_d_p = 1.0 / max(vd_p.z, 1e-5);

            // NeighborMax vector sample at the center point
            const half2 v_max = SAMPLE_TEXTURE2D(_NeighborMaxTex, sampler_LinearClamp, input.texcoord + JitterTile(input.texcoord)).xy;
            const half l_v_max = length(v_max);
            const half rcp_l_v_max = 1.0 / l_v_max;

            // Escape early if the NeighborMax vector is small enough.
            if (l_v_max < 2.0) return c_p;

            // Use V_p as a secondary sampling direction except when it's too small
            // compared to V_max. This vector is rescaled to be the length of V_max.
            const half2 v_alt = (l_v_p * 2.0 > l_v_max) ? vd_p.xy * (l_v_max / l_v_p) : v_max;

            // Determine the sample count.
            const half sc = floor(min(_LoopCount, l_v_max * 0.5));

            // Loop variables (starts from the outermost sample)
            const half dt = 1.0 / sc;
            const half t_offs = (InterleavedGradientNoise(input.texcoord * _SourceSize.xy,0) - 0.5) * dt;
            half t = 1.0 - dt * 0.5;
            half count = 0.0;

            // Background velocity
            // This is used for tracking the maximum velocity in the background layer.
            half l_v_bg = max(l_v_p, 1.0);

            // Color accumlation
            half4 acc = 0.0;

            UNITY_LOOP
            while (t > dt * 0.25)
            {
                // Sampling direction (switched per every two samples)
                const half2 v_s = Interval(count, 4.0) ? v_alt : v_max;

                // Sample position (inverted per every sample)
                const half t_s = (Interval(count, 2.0) ? -t : t) + t_offs;

                // Distance to the sample position
                const half l_t = l_v_max * abs(t_s);

                // UVs for the sample position
                const half2 uv0 = input.texcoord + v_s * t_s * _MotionVectorTexture_TexelSize.xy;
                const half2 uv1 = input.texcoord + v_s * t_s * _MotionVectorTexture_TexelSize.xy;

                // Color sample
                const half3 c = SAMPLE_TEXTURE2D(_BlitTexture, sampler_MirrorLinear, uv0).rgb;

                // Velocity/Depth sample
                const half3 vd = SampleVelocity(uv1);

                // Background/Foreground separation
                const half fg = saturate((vd_p.z - vd.z) * _Separation * rcp_d_p);

                // Length of the velocity vector
                const half l_v = lerp(l_v_bg, length(vd.xy), fg);

                // Sample weight
                // (Distance test) * (Spreading out by motion) * (Triangular window)
                const half w = saturate(l_v - l_t) / (l_v + 0.001f);

                // Color accumulation
                acc += half4(c, 1.0) * w;

                // Update the background velocity.
                l_v_bg = max(l_v_bg, l_v);

                // Advance to the next sample.
                t = Interval(count, 2.0) ? t - dt : t;
                count += 1.0;
            }

            // Add the center sample.
            acc += half4(c_p.rgb, 1.0) * (1.2 / (l_v_bg * sc * 2.0));

            return half4(acc.rgb / (acc.a + 0.001f), c_p.a);
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

                half4 FragCMB(VaryingsCMB input) : SV_Target
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

                half4 FragCMB(VaryingsCMB input) : SV_Target
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

                half4 FragCMB(VaryingsCMB input) : SV_Target
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

                half4 FragCMB(VaryingsCMB input) : SV_Target
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

                half4 FragCMB(VaryingsCMB input) : SV_Target
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

                half4 FragCMB(VaryingsCMB input) : SV_Target
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

                half4 FragCMB(VaryingsCMB input) : SV_Target
                {
                    return FragTileMax4(input);
                }

            ENDHLSL
        }
        
    }
}
