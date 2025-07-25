using System;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering.RenderGraphModule;

namespace UnityEngine.Rendering.HighDefinition
{
    partial class VolumetricCloudsSystem
    {
        RTHandle m_AdvancedCloudMap;
        int m_CloudMapHash;
        int m_EvaluateCloudMapKernel;
        ComputeShader m_CloudMapGeneratorCS;

        void InitializeVolumetricCloudsMap()
        {
            // Grab the kernels we need
            m_CloudMapGeneratorCS = m_RuntimeResources.volumetricCloudMapGeneratorCS;
            m_EvaluateCloudMapKernel = m_CloudMapGeneratorCS.FindKernel("EvaluateCloudMap");
            m_CloudMapHash = 0;
        }

        void ReleaseVolumetricCloudsMap()
        {
            RTHandles.Release(m_AdvancedCloudMap);
        }

        // This function evaluates for the current advanced configuration it hash, for identification, storage and retrieval.
        static int EvaluateCurrentAdvancedHash(in VolumetricClouds settings)
        {
            unchecked
            {
                int hash = HDUtils.GetTextureHash(settings.cumulusMap.value != null ? settings.cumulusMap.value : Texture2D.blackTexture);
                hash = 23 * hash + settings.cumulusMapMultiplier.value.GetHashCode();
                hash = 23 * hash + HDUtils.GetTextureHash(settings.altoStratusMap.value != null ? settings.altoStratusMap.value : Texture2D.blackTexture);
                hash = 23 * hash + settings.altoStratusMapMultiplier.value.GetHashCode();
                hash = 23 * hash + HDUtils.GetTextureHash(settings.cumulonimbusMap.value != null ? settings.cumulonimbusMap.value : Texture2D.blackTexture);
                hash = 23 * hash + settings.cumulonimbusMapMultiplier.value.GetHashCode();
                hash = 23 * hash + HDUtils.GetTextureHash(settings.rainMap.value != null ? settings.rainMap.value : Texture2D.blackTexture);
                hash = 23 * hash + settings.cloudMapResolution.value.GetHashCode();
                return hash;
            }
        }

        bool RequiresCloudMapBaking(HDCamera hdCamera, in VolumetricClouds settings)
        {
            // Evaluate if we need to recompute the texture
            bool status = (HasVolumetricClouds(hdCamera, in settings)
                && settings.cloudControl.value == VolumetricClouds.CloudControl.Advanced);

            if (status)
            {
                // Evaluate the hash of the current configuration
                int currentHash = EvaluateCurrentAdvancedHash(in settings);
                if (m_CloudMapHash == currentHash)
                    status = false;
                else
                    m_CloudMapHash = currentHash;
            }

            return status;
        }

        bool IsWrapModeClamp(TextureParameter tex) => tex.value != null && tex.value.wrapMode == TextureWrapMode.Clamp;

        void AdjustCloudMapTextureSize(in VolumetricClouds settings)
        {
            int cloudMapRes = (int)settings.cloudMapResolution.value;
            bool clamp = IsWrapModeClamp(settings.cumulusMap) || IsWrapModeClamp(settings.altoStratusMap) || IsWrapModeClamp(settings.cumulonimbusMap);

            // Evaluate if a (re)allocation is required
            bool needAllocation = m_AdvancedCloudMap == null;
            if (m_AdvancedCloudMap != null && (m_AdvancedCloudMap.rt.width != cloudMapRes || (m_AdvancedCloudMap.rt.wrapMode == TextureWrapMode.Clamp) != clamp))
            {
                RTHandles.Release(m_AdvancedCloudMap);
                needAllocation = true;
            }

            if (needAllocation)
            {
                var wrapMode = clamp ? TextureWrapMode.Clamp : TextureWrapMode.Repeat;
                m_AdvancedCloudMap = RTHandles.Alloc(cloudMapRes, cloudMapRes, 1, colorFormat: GraphicsFormat.R8G8B8A8_UNorm,
                    enableRandomWrite: true, useDynamicScale: false, useMipMap: false, filterMode: FilterMode.Bilinear, wrapMode: wrapMode, name: "Volumetric Clouds Map");
            }
        }

        struct CloudMapGenerationParameters
        {
            public int cloudMapResolution;

            public ComputeShader generationCS;
            public int generationKernel;

            public Texture cumulusMap;
            public float cumulusMapMultiplier;
            public Texture altostratusMap;
            public float altoStratusMapMultiplier;
            public Texture cumulonimbusMap;
            public float cumulonimbusMapMultiplier;
            public Texture rainMap;
        }

        CloudMapGenerationParameters PrepareCloudMapGenerationParameters(in VolumetricClouds settings)
        {
            CloudMapGenerationParameters parameters = new CloudMapGenerationParameters();

            parameters.cloudMapResolution = (int)settings.cloudMapResolution.value;

            parameters.generationCS = m_CloudMapGeneratorCS;
            parameters.generationKernel = m_EvaluateCloudMapKernel;

            parameters.cumulusMap = settings.cumulusMap.value != null ? settings.cumulusMap.value : Texture2D.blackTexture;
            parameters.cumulusMapMultiplier = settings.cumulusMapMultiplier.value;
            parameters.altostratusMap = settings.altoStratusMap.value != null ? settings.altoStratusMap.value : Texture2D.blackTexture;
            parameters.altoStratusMapMultiplier = settings.altoStratusMapMultiplier.value;
            parameters.cumulonimbusMap = settings.cumulonimbusMap.value != null ? settings.cumulonimbusMap.value : Texture2D.blackTexture;
            parameters.cumulonimbusMapMultiplier = settings.cumulonimbusMapMultiplier.value;
            parameters.rainMap = settings.rainMap.value != null ? settings.rainMap.value : Texture2D.blackTexture;

            return parameters;
        }

        static void EvaluateVolumetricCloudMap(CommandBuffer cmd, CloudMapGenerationParameters parameters, RTHandle outputCloudMap)
        {
            using (new ProfilingScope(cmd, ProfilingSampler.Get(HDProfileId.VolumetricCloudMapGeneration)))
            {
                // Compute the final resolution parameters
                int resolutionRounded = HDUtils.DivRoundUp(parameters.cloudMapResolution, 8);
                cmd.SetComputeIntParam(parameters.generationCS, HDShaderIDs._CloudMapResolution, parameters.cloudMapResolution);

                cmd.SetComputeTextureParam(parameters.generationCS, parameters.generationKernel, HDShaderIDs._CumulusMap, parameters.cumulusMap);
                cmd.SetComputeFloatParam(parameters.generationCS, HDShaderIDs._CumulusMapMultiplier, parameters.cumulusMapMultiplier);

                cmd.SetComputeTextureParam(parameters.generationCS, parameters.generationKernel, HDShaderIDs._AltostratusMap, parameters.altostratusMap);
                cmd.SetComputeFloatParam(parameters.generationCS, HDShaderIDs._AltostratusMapMultiplier, parameters.altoStratusMapMultiplier);

                cmd.SetComputeTextureParam(parameters.generationCS, parameters.generationKernel, HDShaderIDs._CumulonimbusMap, parameters.cumulonimbusMap);
                cmd.SetComputeFloatParam(parameters.generationCS, HDShaderIDs._CumulonimbusMapMultiplier, parameters.cumulonimbusMapMultiplier);

                cmd.SetComputeTextureParam(parameters.generationCS, parameters.generationKernel, HDShaderIDs._RainMap, parameters.rainMap);

                cmd.SetComputeTextureParam(parameters.generationCS, parameters.generationKernel, HDShaderIDs._CloudMapTextureRW, outputCloudMap);
                cmd.DispatchCompute(parameters.generationCS, parameters.generationKernel, resolutionRounded, resolutionRounded, 1);
            }
        }

        class VolumetricCloudsMapData
        {
            public CloudMapGenerationParameters parameters;
            public TextureHandle cloudMapTexture;
        }

        void PreRenderVolumetricCloudMap(RenderGraph renderGraph, HDCamera hdCamera, in VolumetricClouds settings)
        {
            // If we don't need to bake the volumetric cloud map, skip right away
            if (!RequiresCloudMapBaking(hdCamera, settings))
                return;

            // Make sure the cloud map is at the right size
            AdjustCloudMapTextureSize(in settings);

            using (var builder = renderGraph.AddUnsafePass<VolumetricCloudsMapData>("Volumetric cloud map generation", out var passData, ProfilingSampler.Get(HDProfileId.VolumetricCloudMapGeneration)))
            {
                builder.AllowPassCulling(false);

                passData.parameters = PrepareCloudMapGenerationParameters(in settings);
                passData.cloudMapTexture = renderGraph.ImportTexture(m_AdvancedCloudMap);

                builder.SetRenderFunc(
                    (VolumetricCloudsMapData data, UnsafeGraphContext ctx) =>
                    {
                        EvaluateVolumetricCloudMap(CommandBufferHelpers.GetNativeCommandBuffer(ctx.cmd), data.parameters, data.cloudMapTexture);
                    });
            }
        }
    }
}
