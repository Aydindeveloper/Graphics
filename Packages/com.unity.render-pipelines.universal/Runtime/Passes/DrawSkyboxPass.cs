using System;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering.Universal.Internal;

namespace UnityEngine.Rendering.Universal
{
    /// <summary>
    /// Draw the skybox into the given color buffer using the given depth buffer for depth testing.
    ///
    /// This pass renders the standard Unity skybox.
    /// </summary>
    public partial class DrawSkyboxPass : ScriptableRenderPass
    {
        /// <summary>
        /// Creates a new <c>DrawSkyboxPass</c> instance.
        /// </summary>
        /// <param name="evt">The <c>RenderPassEvent</c> to use.</param>
        /// <seealso cref="RenderPassEvent"/>
        public DrawSkyboxPass(RenderPassEvent evt)
        {
            profilingSampler = ProfilingSampler.Get(URPProfileId.DrawSkybox);
            renderPassEvent = evt;
        }

#if URP_COMPATIBILITY_MODE
        /// <inheritdoc/>
        [Obsolete(DeprecationMessage.CompatibilityScriptingAPIObsoleteFrom2023_3)]
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            UniversalCameraData cameraData = renderingData.frameData.Get<UniversalCameraData>();
            var activeDebugHandler = GetActiveDebugHandler(cameraData);
            if (activeDebugHandler != null)
            {
                // TODO: The skybox needs to work the same as the other shaders, but until it does we'll not render it
                // when certain debug modes are active (e.g. wireframe/overdraw modes)
                if (activeDebugHandler.IsScreenClearNeeded)
                {
                    return;
                }
            }

            var skyRendererList = CreateSkyboxRendererList(context, cameraData);
            ExecutePass(CommandBufferHelpers.GetRasterCommandBuffer(renderingData.commandBuffer), cameraData.xr, skyRendererList);
        }

        private RendererList CreateSkyboxRendererList(ScriptableRenderContext context, UniversalCameraData cameraData)
        {
            var skyRendererList = new RendererList();

#if ENABLE_VR && ENABLE_XR_MODULE
            if (cameraData.xr.enabled)
            {
                // Setup Legacy XR buffer states
                if (cameraData.xr.singlePassEnabled)
                {
                    skyRendererList = context.CreateSkyboxRendererList(cameraData.camera,
                        cameraData.GetProjectionMatrix(0), cameraData.GetViewMatrix(0),
                        cameraData.GetProjectionMatrix(1), cameraData.GetViewMatrix(1));
                }
                else
                {
                    skyRendererList = context.CreateSkyboxRendererList(cameraData.camera, cameraData.GetProjectionMatrix(0), cameraData.GetViewMatrix(0));
                }
            }
            else
#endif
            {
                skyRendererList = context.CreateSkyboxRendererList(cameraData.camera);
            }

            return skyRendererList;
        }
#endif

        private RendererListHandle CreateSkyBoxRendererList(RenderGraph renderGraph, UniversalCameraData cameraData)
        {
            var skyRendererListHandle = new RendererListHandle();

#if ENABLE_VR && ENABLE_XR_MODULE
            if (cameraData.xr.enabled)
            {
                // Setup Legacy XR buffer states
                if (cameraData.xr.singlePassEnabled)
                {
                    skyRendererListHandle = renderGraph.CreateSkyboxRendererList(cameraData.camera,
                        cameraData.GetProjectionMatrix(0), cameraData.GetViewMatrix(0),
                        cameraData.GetProjectionMatrix(1), cameraData.GetViewMatrix(1));
                }
                else
                {
                    skyRendererListHandle = renderGraph.CreateSkyboxRendererList(cameraData.camera, cameraData.GetProjectionMatrix(0), cameraData.GetViewMatrix(0));
                }
            }
            else
#endif
            {
                skyRendererListHandle = renderGraph.CreateSkyboxRendererList(cameraData.camera,cameraData.GetProjectionMatrix(0),cameraData.GetViewMatrix(0));
            }

            return skyRendererListHandle;
        }

        private static void ExecutePass(RasterCommandBuffer cmd, XRPass xr, RendererList rendererList)
        {
#if ENABLE_VR && ENABLE_XR_MODULE
            if (xr.enabled && xr.singlePassEnabled)
                cmd.SetSinglePassStereo(SystemInfo.supportsMultiview ? SinglePassStereoMode.Multiview : SinglePassStereoMode.Instancing);
#endif
            cmd.DrawRendererList(rendererList);

#if ENABLE_VR && ENABLE_XR_MODULE
            if (xr.enabled && xr.singlePassEnabled)
                cmd.SetSinglePassStereo(SinglePassStereoMode.None);
#endif
        }

        private class PassData
        {
            internal XRPass xr;
            internal RendererListHandle skyRendererListHandle;
            internal Material material;
        }

        private void InitPassData(ref PassData passData, in XRPass xr, in RendererListHandle handle)
        {
            passData.xr = xr;
            passData.skyRendererListHandle = handle;
        }

        internal void Render(RenderGraph renderGraph, ContextContainer frameData, ScriptableRenderContext context, TextureHandle colorTarget, TextureHandle depthTarget, Material skyboxMaterial)
        {
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();

            var activeDebugHandler = GetActiveDebugHandler(cameraData);
            if (activeDebugHandler != null)
            {
                // TODO: The skybox needs to work the same as the other shaders, but until it does we'll not render it
                // when certain debug modes are active (e.g. wireframe/overdraw modes)
                if (activeDebugHandler.IsScreenClearNeeded)
                {
                    return;
                }
            }

            using (var builder = renderGraph.AddRasterRenderPass<PassData>(passName, out var passData, profilingSampler))
            {
                var skyRendererListHandle = CreateSkyBoxRendererList(renderGraph, cameraData);
                InitPassData(ref passData, cameraData.xr, skyRendererListHandle);
                passData.material = skyboxMaterial;
                builder.UseRendererList(skyRendererListHandle);
                builder.SetRenderAttachment(colorTarget, 0, AccessFlags.Write);
                builder.SetRenderAttachmentDepth(depthTarget, AccessFlags.Write);

                builder.AllowPassCulling(false);
                if (cameraData.xr.enabled)
                {
                    bool passSupportsFoveation = cameraData.xrUniversal.canFoveateIntermediatePasses || resourceData.isActiveTargetBackBuffer;
                    builder.EnableFoveatedRasterization(cameraData.xr.supportsFoveatedRendering && passSupportsFoveation);
                }

                builder.SetRenderFunc((PassData data, RasterGraphContext context) =>
                {
                    ExecutePass(context.cmd, data.xr, data.skyRendererListHandle);
                });
            }
        }
    }
}
