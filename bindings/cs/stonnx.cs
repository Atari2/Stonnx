using System;
using System.Runtime.InteropServices;
using System.Text;

namespace StonnxCLR {
    public static unsafe class StonnxAPI {
        [DllImport("stonnx_api", EntryPoint = "read_onnx_model", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern IntPtr _read_onnx_model(string model_path);

        [DllImport("stonnx_api", EntryPoint = "free_onnx_model", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern void _free_onnx_model(IntPtr model);

        [DllImport("stonnx_api", EntryPoint = "get_opset_version", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.I8)]
        private static extern Int64 _get_opset_version(IntPtr model);

        [DllImport("stonnx_api", EntryPoint = "run_model", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool _run_model(string model_path, Int64 verbosity, Int64 graph_format, Int64 failfast);

        [DllImport("stonnx_api", EntryPoint = "last_error", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern sbyte* _last_error();

        public abstract class PointerInternalBase : IDisposable
        {
            protected readonly IntPtr data_pointer;
            protected bool disposedValue;

            protected PointerInternalBase(IntPtr ptr)
            {
                data_pointer = ptr;
            }

            protected virtual void Dispose(bool disposing)
            {
                // Do nothing as base case, subclasses will override and do things if necessary
            }
            ~PointerInternalBase()
            {
                Dispose(disposing: false);
            }

            public void Dispose()
            {
                Dispose(disposing: true);
                GC.SuppressFinalize(this);
            }
        }

        public enum ExecutionMode : Int64
        {
            FailFast = 1,
            Continue = 0
        }

        public enum GraphFormat : Int64
        {
            None = 0,
            Json = 1,
            Dot = 2
        }

        public enum Verbosity : Int64
        {
            Minimal = 0,
            Informational = 1,
            Results = 2,
            Intermediate = 4
        }

        public class ONNXModel : PointerInternalBase
        {
            private ONNXModel(IntPtr ptr) : base(ptr) { }
            protected override void Dispose(bool disposing)
            {
                if (!disposedValue)
                {
                    if (disposing)
                    {

                    }
                    _free_onnx_model(data_pointer);
                    disposedValue = true;
                }
            }
            public static bool Run(string modelpath, Verbosity verbosity, GraphFormat graph_format, ExecutionMode execution_mode)
            {
                return _run_model(modelpath, (Int64)verbosity, (Int64)graph_format, (Int64)execution_mode);
            }
            #nullable enable
            public static ONNXModel? FromFile(string modelpath)
            {
                IntPtr ptr = _read_onnx_model(modelpath);
                if (ptr == IntPtr.Zero)
                {
                    return null;
                } else
                {
                    return new ONNXModel(ptr);
                }
            }
            #nullable disable
            public int GetOpsetVersion()
            {
                Int64 opsetVersion = _get_opset_version(data_pointer);
                return (int)opsetVersion;
            }
        }

        public static string LastError()
        {
            var cstr = _last_error();
            string str = new(cstr);
            return str;
        }
    }
}