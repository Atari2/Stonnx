using System;
using ONNXRustProtoCLR;

public class HelloWorld {
    public static int Main() 
    {
        if (ONNXRustProtoAPI.ONNXModel.Run("GPT2", ONNXRustProtoAPI.Verbosity.Minimal, ONNXRustProtoAPI.GraphFormat.None, ONNXRustProtoAPI.ExecutionMode.FailFast)) {
            Console.WriteLine("Model execution succeeded");
            return 0;
        } else {
            Console.WriteLine("Model execution failed");
            Console.WriteLine("Error: " + ONNXRustProtoAPI.LastError());
            return 1;
        }
    }
}