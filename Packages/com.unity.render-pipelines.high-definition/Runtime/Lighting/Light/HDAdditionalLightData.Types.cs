using System;
using System.Linq;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace UnityEngine.Rendering.HighDefinition
{
    /// <summary>Shadow Update mode </summary>
    public enum ShadowUpdateMode
    {
        /// <summary>Shadow map will be rendered at every frame.</summary>
        EveryFrame = 0,
        /// <summary>Shadow will be rendered only when the OnEnable of the light is called.</summary>
        OnEnable,
        /// <summary>Shadow will be rendered when you call HDAdditionalLightData.RequestShadowMapRendering().</summary>
        OnDemand
    }

    /// <summary>
    /// Options for defining which RenderingLayers affect which renderers.
    /// </summary>
    /// <remarks>
    /// This enum is a bitfield, which means that you can combine multiple values using the bitwise OR operator (|).
    /// </remarks>
    /// <example>
    /// <code lang="cs"><![CDATA[
    /// using UnityEngine;
    /// using RenderingLayerMask = UnityEngine.Rendering.HighDefinition.RenderingLayerMask;
    ///
    /// public class RenderingLayerMaskExample : MonoBehaviour
    /// {
    ///     void Start()
    ///     {
    ///         // Set the rendering layers for the renderer
    ///         // The layers are RenderingLayer1 and RenderingLayer2,
    ///         // so the renderer is affected only by lights and effects in those layers.
    ///
    ///         RenderingLayerMask desiredLayerMask = RenderingLayerMask.RenderingLayer1 | RenderingLayerMask.RenderingLayer2;
    ///         Renderer renderer = GetComponent<Renderer>();
    ///         if (renderer != null)
    ///         {
    ///             // If the Renderer component exists, we set the desired rendering layer mask
    ///             renderer.renderingLayerMask = (uint)desiredLayerMask;
    ///         }
    ///     }
    /// }
    /// ]]></code>
    /// </example>
    [Flags]
    public enum RenderingLayerMask
    {
        /// <summary>No rendering layer.</summary>
        Nothing = 0,
        /// <summary>Rendering layer 1.</summary>
        RenderingLayer1 = 1 << 0,
        /// <summary>Rendering layer 2.</summary>
        RenderingLayer2 = 1 << 1,
        /// <summary>Rendering layer 3.</summary>
        RenderingLayer3 = 1 << 2,
        /// <summary>Rendering layer 4.</summary>
        RenderingLayer4 = 1 << 3,
        /// <summary>Rendering layer 5.</summary>
        RenderingLayer5 = 1 << 4,
        /// <summary>Rendering layer 6.</summary>
        RenderingLayer6 = 1 << 5,
        /// <summary>Rendering layer 7.</summary>
        RenderingLayer7 = 1 << 6,
        /// <summary>Rendering layer 8.</summary>
        RenderingLayer8 = 1 << 7,
        /// <summary>Rendering layer 9.</summary>
        RenderingLayer9 = 1 << 8,
        /// <summary>Rendering layer 10.</summary>
        RenderingLayer10 = 1 << 9,
        /// <summary>Rendering layer 11.</summary>
        RenderingLayer11 = 1 << 10,
        /// <summary>Rendering layer 12.</summary>
        RenderingLayer12 = 1 << 11,
        /// <summary>Rendering layer 13.</summary>
        RenderingLayer13 = 1 << 12,
        /// <summary>Rendering layer 14.</summary>
        RenderingLayer14 = 1 << 13,
        /// <summary>Rendering layer 15.</summary>
        RenderingLayer15 = 1 << 14,
        /// <summary>Rendering layer 16.</summary>
        RenderingLayer16 = 1 << 15,

        /// <summary>Default Layer for lights.</summary>
        [HideInInspector, Obsolete("Use UnityEngine.RenderingLayerMask.defaultRenderingLayerMask instead. #from(2023.1) ")]
        LightLayerDefault = RenderingLayer1,
        /// <summary>Default Layer for decals.</summary>
        [HideInInspector, Obsolete("Use UnityEngine.RenderingLayerMask.defaultRenderingLayerMask instead. #from(2023.1) ")]
        DecalLayerDefault = RenderingLayer9,
        /// <summary>Default rendering layers mask.</summary>
        [HideInInspector, Obsolete("Use UnityEngine.RenderingLayerMask.defaultRenderingLayerMask instead. #from(2023.1) ")]
        Default = LightLayerDefault | DecalLayerDefault,
        /// <summary>All layers enabled.</summary>
        [HideInInspector]
        Everything = 0xFFFF,


        /// <summary>Light Layer 1.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer2. #from(2023.1)")]
        LightLayer1 = RenderingLayer2,
        /// <summary>Light Layer 2.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer3. #from(2023.1)")]
        LightLayer2 = RenderingLayer3,
        /// <summary>Light Layer 3.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer4. #from(2023.1)")]
        LightLayer3 = RenderingLayer4,
        /// <summary>Light Layer 4.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer5. #from(2023.1)")]
        LightLayer4 = RenderingLayer5,
        /// <summary>Light Layer 5.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer6. #from(2023.1)")]
        LightLayer5 = RenderingLayer6,
        /// <summary>Light Layer 6.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer7. #from(2023.1)")]
        LightLayer6 = RenderingLayer7,
        /// <summary>Light Layer 7.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer8. #from(2023.1)")]
        LightLayer7 = RenderingLayer8,

        /// <summary>Decal Layer 1.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer10. #from(2023.1)")]
        DecalLayer1 = RenderingLayer10,
        /// <summary>Decal Layer 2.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer11. #from(2023.1)")]
        DecalLayer2 = RenderingLayer11,
        /// <summary>Decal Layer 3.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer12. #from(2023.1)")]
        DecalLayer3 = RenderingLayer12,
        /// <summary>Decal Layer 4.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer13. #from(2023.1)")]
        DecalLayer4 = RenderingLayer13,
        /// <summary>Decal Layer 5.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer14. #from(2023.1)")]
        DecalLayer5 = RenderingLayer14,
        /// <summary>Decal Layer 6.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer15. #from(2023.1)")]
        DecalLayer6 = RenderingLayer15,
        /// <summary>Decal Layer 7.</summary>
        [HideInInspector, Obsolete("Use RenderingLayer16. #from(2023.1)")]
        DecalLayer7 = RenderingLayer16,
    }

    /// <summary>
    /// Extension class for the HDLightTypeAndShape type.
    /// </summary>
    public static class HDLightTypeExtension
    {
        /// <summary>
        /// Returns true if the light type is a spot light
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool IsSpot(this LightType type)
            => type == LightType.Box
            || type == LightType.Pyramid
            || type == LightType.Spot;

        /// <summary>
        /// Returns true if the light type is an area light
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool IsArea(this LightType type)
            => type == LightType.Tube
            || type == LightType.Rectangle
            || type == LightType.Disc;

        /// <summary>
        /// Returns true if the light type can be used for runtime lighting
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool SupportsRuntimeOnly(this LightType type)
            => type != LightType.Disc;

        /// <summary>
        /// Returns true if the light type can be used for baking
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool SupportsBakedOnly(this LightType type)
            => type != LightType.Tube;

        /// <summary>
        /// Returns true if the light type can be used in mixed mode
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool SupportsMixed(this LightType type)
            => type != LightType.Tube
            && type != LightType.Disc;
    }
}
