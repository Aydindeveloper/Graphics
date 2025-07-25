using System;
using System.Linq;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.VFX;
using UnityObject = UnityEngine.Object;

namespace UnityEditor.VFX
{
#pragma warning disable 0659
    abstract class VFXValue : VFXExpression
    {
        public enum Mode
        {
            Variable, // Variable that should never be folded
            FoldableVariable, // Variable that can be folded
            Constant, // Immutable value
        }

        // Syntactic sugar method to create a constant value

        static public VFXValue<EntityId> Constant(Texture2D value)
        {
            return new VFXTexture2DValue(ReferenceEquals(value, null) ? EntityId.None : value.GetEntityId(), Mode.Constant);
        }

        static public VFXValue<EntityId> Constant(Texture3D value)
        {
            return new VFXTexture3DValue(ReferenceEquals(value, null) ? EntityId.None : value.GetEntityId(), Mode.Constant);
        }

        static public VFXValue<EntityId> Constant(Cubemap value)
        {
            return new VFXTextureCubeValue(ReferenceEquals(value, null) ? EntityId.None : value.GetEntityId(), Mode.Constant);
        }

        static public VFXValue<EntityId> Constant(Texture2DArray value)
        {
            return new VFXTexture2DArrayValue(ReferenceEquals(value, null) ? EntityId.None : value.GetEntityId(), Mode.Constant);
        }

        static public VFXValue<EntityId> Constant(CubemapArray value)
        {
            return new VFXTextureCubeArrayValue(ReferenceEquals(value, null) ? EntityId.None : value.GetEntityId(), Mode.Constant);
        }

        static public VFXValue<EntityId> Constant(CameraBuffer value)
        {
            return new VFXCameraBufferValue(value, Mode.Constant);
        }

        static public VFXValue<T> Constant<T>(T value = default(T))
        {
            return new VFXValue<T>(value, Mode.Constant);
        }

        private static Flags GetFlagsFromMode(Mode mode, Flags flags)
        {
            flags |= Flags.Value;
            if (mode != Mode.Variable)
            {
                flags |= Flags.Foldable;
                if (mode == Mode.Constant)
                    flags |= Flags.Constant;
            }
            return flags;
        }

        protected VFXValue(Mode mode, Flags flags)
            : base(GetFlagsFromMode(mode, flags))
        {
            m_Mode = mode;
        }

        public Mode ValueMode { get { return m_Mode; } }

        sealed public override VFXExpressionOperation operation { get { return VFXExpressionOperation.Value; } }

        protected sealed override VFXExpression Evaluate(VFXExpression[] constParents)
        {
            if (m_Mode == Mode.Constant)
                return this;

            return CopyExpression(Mode.Constant);
        }

        public override string GetCodeString(string[] parents)
        {
            if (Is(Flags.InvalidOnGPU) || !Is(Flags.Constant))
                throw new InvalidOperationException(string.Format("Type {0} is either not valid on GPU or expression is not constant", valueType));

            return VFXShaderWriter.GetValueString(valueType, GetContent());
        }

        abstract public VFXValue CopyExpression(Mode mode);

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(this, obj))
                return true;

            if (m_Mode == Mode.Constant)
            {
                var val = obj as VFXValue;
                if (val == null)
                    return false;

                if (val.m_Mode != Mode.Constant)
                    return false;

                if (valueType != val.valueType)
                    return false;

                var content = GetContent();
                var otherContent = val.GetContent();

                if (content == null)
                    return otherContent == null;

                return content.Equals(otherContent);
            }
            else
                return false;
        }

        protected override int GetInnerHashCode()
        {
            if (m_Mode == Mode.Constant)
            {
                int hashCode = valueType.GetHashCode();
                var content = GetContent();

                if (content != null)
                    hashCode ^= content.GetHashCode();

                return hashCode;
            }

            return RuntimeHelpers.GetHashCode(this);
        }

        public abstract void SetContent(object value);

        private Mode m_Mode;
        protected object m_Content;
    }

    class VFXValue<T> : VFXValue
    {
        protected static Flags GetFlagsFromType(VFXValueType valueType)
        {
            var flags = Flags.None;
            if (!IsTypeValidOnGPU(valueType))
                flags |= VFXExpression.Flags.InvalidOnGPU;
            if (!IsTypeConstantFoldable(valueType))
                flags |= VFXExpression.Flags.InvalidConstant;
            return flags;
        }

        public VFXValue(T content, Mode mode = Mode.FoldableVariable, Flags flag = Flags.None) : base(mode, flag | GetFlagsFromType(ToValueType()))
        {
            m_Content = content;
        }

        public override VFXValue CopyExpression(Mode mode)
        {
            var copy = new VFXValue<T>(Get(), mode);
            return copy;
        }

        private static readonly VFXValue s_Default = new VFXValue<T>(default(T), VFXValue.Mode.Constant);
        public static VFXValue Default { get { return s_Default; } }

        public T Get() => (T)m_Content;

        public override object GetContent()
        {
            return Get();
        }

        public override void SetContent(object value)
        {
            m_Content = default(T);
            if (value == null)
            {
                return;
            }

            var fromType = value.GetType();
            var toType = typeof(T);

            if (typeof(Texture).IsAssignableFrom(toType) && toType.IsAssignableFrom(fromType))
            {
                m_Content = (T)value;
            }
            else if (typeof(Mesh).IsAssignableFrom(toType) && toType.IsAssignableFrom(fromType))
            {
                m_Content = (T)value;
            }
            else if (fromType == toType || toType.IsAssignableFrom(fromType))
            {
                m_Content = (T)Convert.ChangeType(value, toType);
            }
            else if (toType == typeof(GraphicsBuffer))
            {
                //We can't serialize a reference of GraphicsBuffer
                m_Content = null;
            }
            else
            {
                var implicitMethod = fromType.GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
                    .FirstOrDefault(m => m.Name == "op_Implicit" && m.ReturnType == toType);
                if (implicitMethod != null)
                {
                    m_Content = (T)implicitMethod.Invoke(null, new object[] { value });
                }
                else
                {
                    Debug.LogErrorFormat("Cannot cast from {0} to {1}", fromType, toType);
                }
            }
        }

        private static VFXValueType ToValueType()
        {
            Type t = typeof(T);
            if (typeof(Texture).IsAssignableFrom(t))
            {
                return VFXValueType.None;
            }
            var valueType = GetVFXValueTypeFromType(t);
            if (valueType == VFXValueType.None)
                throw new ArgumentException($"Invalid type for {t}");
            return valueType;
        }

        static private readonly VFXValueType s_ValueType = ToValueType();
        protected override int[] additionnalOperands
        {
            get
            {
                return new int[] { (int)s_ValueType };
            }
        }
    }

    class VFXObjectValue : VFXValue<EntityId>
    {
        public VFXObjectValue(EntityId entityId, Mode mode, VFXValueType contentType) : base(entityId, mode, GetFlagsFromType(contentType))
        {
            m_ContentType = contentType;
        }

        sealed protected override int[] additionnalOperands
        {
            get
            {
                return new int[] { (int)m_ContentType };
            }
        }

        public override VFXValue CopyExpression(Mode mode)
        {
            var copy = new VFXObjectValue((EntityId)m_Content, mode, m_ContentType);
            return copy;
        }

        public override T Get<T>()
        {
            if (typeof(T) == typeof(EntityId))
                return (T)(object)base.Get();
            Debug.Assert(UnsafeUtility.SizeOf<EntityId>() == sizeof(int), "EntityId size is not equal to int size, this will cause issues, update to VFXValue<EntityId> instead");

            return (T)(object)EditorUtility.EntityIdToObject(base.Get());
        }

        public override object GetContent()
        {
            return Get();
        }

        public override void SetContent(object value)
        {
            if (value == null)
            {
                m_Content = EntityId.None;
                return;
            }
            if (value is UnityObject obj)
            {
                m_Content = obj.GetEntityId();
                return;
            }
            if (value is CameraBuffer cameraBuffer)
            {
                m_Content = cameraBuffer;
                return;
            }
            if (value is GraphicsBuffer)
            {
                m_Content = EntityId.None;
                return;
            }

            m_Content = (EntityId)value;
        }

        VFXValueType m_ContentType;
    }

#pragma warning restore 0659
}
